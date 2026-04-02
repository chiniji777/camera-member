"""Voice Command Processor — Listen to camera audio and trigger smart home actions.

Listens to RTSP audio via FFmpeg, runs Whisper STT every few seconds,
and matches Thai commands to gate control actions.
"""
import subprocess
import threading
import tempfile
import time
import logging
import shutil
import os
import wave
import struct
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Commands: keyword → (channel, label)
GATE_COMMANDS = {
    "เปิดประตู": 0,   # CH1 = Open
    "หยุดประตู": 1,   # CH2 = Hold/Stop
    "ปิดประตู": 2,    # CH3 = Close
}

# Alternative phrases that map to same commands
COMMAND_ALIASES = {
    "เปิด ประตู": 0,
    "หยุด ประตู": 1,
    "ปิด ประตู": 2,
    "เปิดเลย": 0,
    "ปิดเลย": 2,
}

# English commands
ENGLISH_COMMANDS = {
    "open door": 0,
    "open the door": 0,
    "hold door": 1,
    "hold the door": 1,
    "stop door": 1,
    "close door": 2,
    "close the door": 2,
}


def _find_ffmpeg() -> Optional[str]:
    path = shutil.which("ffmpeg")
    if path:
        return path
    fallback = Path("/opt/homebrew/bin/ffmpeg")
    if fallback.exists():
        return str(fallback)
    return None


class VoiceCommandProcessor:
    """Continuously listens to camera audio and detects voice commands."""

    def __init__(
        self,
        on_command: Callable[[int, str], None],
        chunk_seconds: int = 4,
        model_name: str = "tiny",
    ):
        self._on_command = on_command
        self._chunk_seconds = chunk_seconds
        self._model_name = model_name
        self._running = False
        self._processes: dict[str, subprocess.Popen] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._model = None
        self._model_lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            import whisper
            logger.info(f"Loading Whisper model '{self._model_name}'...")
            self._model = whisper.load_model(self._model_name)
            logger.info("Whisper model loaded")

    def start(self, cam_id: str, rtsp_url: str):
        """Start listening on a camera's audio stream."""
        if cam_id in self._threads:
            return

        self._running = True
        t = threading.Thread(
            target=self._listen_loop, args=(cam_id, rtsp_url), daemon=True
        )
        self._threads[cam_id] = t
        t.start()
        logger.info(f"Voice command listening started for {cam_id}")

    def _listen_loop(self, cam_id: str, rtsp_url: str):
        """Main loop: extract audio chunks and run STT."""
        self._load_model()
        ffmpeg_path = _find_ffmpeg()
        if not ffmpeg_path:
            logger.error("FFmpeg not found for voice commands")
            return

        while self._running and cam_id in self._threads:
            try:
                # Record a chunk of audio to a temp WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name

                cmd = [
                    ffmpeg_path,
                    "-rtsp_transport", "tcp",
                    "-i", rtsp_url,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-t", str(self._chunk_seconds),
                    "-y",
                    tmp_path,
                ]

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self._chunk_seconds + 10,
                )

                if proc.returncode != 0:
                    logger.warning(f"FFmpeg audio capture failed for {cam_id}")
                    time.sleep(2)
                    continue

                # Check file has audio data
                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1000:
                    os.unlink(tmp_path)
                    time.sleep(1)
                    continue

                # Skip if audio is too quiet (ambient noise only)
                if not self._has_speech(tmp_path):
                    os.unlink(tmp_path)
                    continue

                # Run Whisper STT
                text = self._transcribe(tmp_path)
                os.unlink(tmp_path)

                if text:
                    text = text.strip()
                    logger.info(f"[Voice {cam_id}] Heard: {text}")
                    self._match_command(text)

            except Exception as e:
                logger.error(f"Voice command error for {cam_id}: {e}")
                time.sleep(2)

    def _has_speech(self, audio_path: str, threshold: int = 1500) -> bool:
        """Check if audio has enough energy to be speech (not just silence/noise)."""
        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                if len(frames) < 100:
                    return False
                samples = struct.unpack(f"<{len(frames)//2}h", frames)
                rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
                logger.debug(f"Audio RMS: {rms:.0f} (threshold: {threshold})")
                return rms > threshold
        except Exception:
            return False

    def _transcribe(self, audio_path: str) -> str:
        """Run Whisper on audio file, return text."""
        try:
            result = self._model.transcribe(
                audio_path,
                language="th",
                fp16=False,
                no_speech_threshold=0.5,
                logprob_threshold=-0.3,
                compression_ratio_threshold=1.5,
                initial_prompt="เปิดประตู หยุดประตู ปิดประตู open door close door hold door",
            )
            # Filter out hallucinations: skip if segments have high no_speech_prob
            segments = result.get("segments", [])
            texts = []
            for seg in segments:
                if seg.get("no_speech_prob", 1.0) < 0.4:
                    texts.append(seg.get("text", ""))
            text = "".join(texts)
            # Reject repetitive hallucinations (e.g. "สาสาสาสาสา")
            if text and self._is_repetitive(text):
                logger.debug(f"Rejected repetitive text: {text[:50]}")
                return ""
            return text
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return ""

    def _is_repetitive(self, text: str, min_len: int = 10) -> bool:
        """Detect repetitive hallucination patterns like 'สาสาสาสา'."""
        clean = text.replace(" ", "")
        if len(clean) < min_len:
            return False
        # Check if any 2-4 char pattern repeats more than 5 times
        for plen in range(1, 5):
            for start in range(min(3, len(clean))):
                pat = clean[start:start + plen]
                if not pat:
                    continue
                count = clean.count(pat)
                if count >= 5 and (count * len(pat)) > len(clean) * 0.5:
                    return True
        return False

    def _match_command(self, text: str):
        """Check if transcribed text matches any gate command (Thai or English)."""
        text_stripped = text.strip()
        if not text_stripped:
            return

        # Check Thai commands (remove spaces for Thai matching)
        text_clean = text_stripped.replace(" ", "")
        thai_chars = sum(1 for c in text_clean if "\u0e00" <= c <= "\u0e7f")

        earliest_pos = len(text_clean) + 1
        best_channel = None
        best_label = None

        if thai_chars >= 3:
            all_commands = {**GATE_COMMANDS}
            for phrase, channel in COMMAND_ALIASES.items():
                all_commands[phrase.replace(" ", "")] = channel

            for phrase, channel in all_commands.items():
                pos = text_clean.find(phrase)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    best_channel = channel
                    best_label = [k for k, v in GATE_COMMANDS.items() if v == channel][0]

        # Check English commands (case-insensitive, keep spaces)
        text_lower = text_stripped.lower()
        for phrase, channel in ENGLISH_COMMANDS.items():
            pos = text_lower.find(phrase)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                best_channel = channel
                best_label = [k for k, v in GATE_COMMANDS.items() if v == channel][0]

        if best_channel is not None:
            logger.info(f"[Voice] Command matched: {best_label} (channel {best_channel})")
            try:
                self._on_command(best_channel, best_label)
            except Exception as e:
                logger.error(f"Command callback error: {e}")

    def stop(self, cam_id: str = None):
        """Stop listening on a camera or all cameras."""
        if cam_id:
            self._threads.pop(cam_id, None)
        else:
            self._running = False
            self._threads.clear()

    def stop_all(self):
        self._running = False
        self._threads.clear()
