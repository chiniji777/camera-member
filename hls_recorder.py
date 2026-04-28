"""HLS recorder — spawns ffmpeg as a subprocess to write rolling .ts segments.

Why a subprocess instead of launchd?
-------------------------------------
macOS LaunchAgents are sandboxed from local-network access by default; ffmpeg
launched from launchctl gets ``No route to host`` when reaching the camera at
192.168.1.x. Spawning from this process inherits the GUI-session permission
that the camera-member server already holds.
"""
from __future__ import annotations
import asyncio
import logging
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
RECORD_ROOT = Path.home() / "recordings"


class HLSRecorder:
    def __init__(self, cam_id: str, rtsp_url: str, bitrate: str = "400k"):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.bitrate = bitrate
        self.dir = RECORD_ROOT / cam_id
        self.seg_dir = self.dir / "segments"
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self.proc: asyncio.subprocess.Process | None = None
        self._stopping = False
        self._task: asyncio.Task | None = None

    def _ffmpeg_args(self) -> list[str]:
        seg_pattern = str(self.seg_dir / "%Y%m%d_%H%M%S_%%05d.ts")
        master = str(self.dir / "index.m3u8")
        return [
            FFMPEG,
            "-loglevel", "warning",
            "-nostats",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-profile:v", "main",
            "-b:v", self.bitrate,
            "-maxrate", "600k",
            "-bufsize", "1200k",
            "-g", "60",
            "-keyint_min", "60",
            "-sc_threshold", "0",
            "-an",
            "-f", "hls",
            "-hls_time", "6",
            "-hls_list_size", "0",
            "-hls_flags", "append_list+independent_segments+omit_endlist+second_level_segment_index",
            "-hls_segment_type", "mpegts",
            "-strftime", "1",
            "-hls_segment_filename", seg_pattern,
            master,
        ]

    async def start(self):
        self._stopping = False
        self._task = asyncio.create_task(self._supervise())
        logger.info(f"[hls] recorder started: {self.cam_id} → {self.dir}")

    async def stop(self):
        self._stopping = True
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.proc.kill()
        if self._task:
            self._task.cancel()
        logger.info(f"[hls] recorder stopped: {self.cam_id}")

    async def _supervise(self):
        backoff = 5
        while not self._stopping:
            try:
                self.proc = await asyncio.create_subprocess_exec(
                    *self._ffmpeg_args(),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                logger.info(f"[hls] ffmpeg pid={self.proc.pid} cam={self.cam_id}")
                stderr_task = asyncio.create_task(self._drain_stderr())
                rc = await self.proc.wait()
                stderr_task.cancel()
                if self._stopping:
                    return
                logger.warning(f"[hls] ffmpeg exited rc={rc} — restart in {backoff}s")
            except Exception as e:
                logger.warning(f"[hls] supervise error: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _drain_stderr(self):
        if not self.proc or not self.proc.stderr:
            return
        try:
            async for line in self.proc.stderr:
                msg = line.decode(errors="replace").rstrip()
                if msg:
                    logger.info(f"[hls/{self.cam_id}] {msg}")
        except asyncio.CancelledError:
            pass

    # ---- Cleanup ----

    def cleanup_old(self, keep_days: int = 7) -> int:
        """Delete .ts segments older than keep_days. Returns count removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        removed = 0
        for p in self.seg_dir.glob("*.ts"):
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    p.unlink()
                    removed += 1
            except FileNotFoundError:
                pass
        if removed:
            logger.info(f"[hls/{self.cam_id}] cleanup removed {removed} old segments")
        return removed

    # ---- Playlist generation ----

    def list_segments(self):
        """Return sorted list of (Path, mtime, duration) — duration assumed 6s."""
        out = []
        for p in sorted(self.seg_dir.glob("*.ts")):
            try:
                out.append((p, p.stat().st_mtime, 6.0))
            except FileNotFoundError:
                pass
        return out

    def live_m3u8(self, window_seconds: int = 60) -> str:
        """Sliding-window live playlist.

        MEDIA-SEQUENCE must reflect the absolute index of the first segment —
        AVPlayer/Safari rely on it to detect playlist updates. The recorder
        names segments YYYYMMDD_HHMMSS_NNNNN.ts where NNNNN is ffmpeg's
        monotonic 5-digit counter.
        """
        segs = self.list_segments()
        if not segs:
            return ""
        needed = max(3, window_seconds // 6 + 2)
        recent = segs[-needed:]

        def seq_of(p):
            try:
                return int(p.stem.rsplit("_", 1)[-1])
            except ValueError:
                return 0

        first_seq = seq_of(recent[0][0])
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:6",
            "#EXT-X-TARGETDURATION:7",
            f"#EXT-X-MEDIA-SEQUENCE:{first_seq}",
            "#EXT-X-INDEPENDENT-SEGMENTS",
        ]
        for p, _, dur in recent:
            lines.append(f"#EXTINF:{dur:.3f},")
            lines.append(p.name)
        return chr(10).join(lines) + chr(10)


    def archive_m3u8(self, since: datetime | None = None, until: datetime | None = None) -> str:
        """Full archive playlist (VOD-style with ENDLIST)."""
        segs = self.list_segments()
        if since:
            segs = [s for s in segs if datetime.fromtimestamp(s[1], tz=timezone.utc) >= since]
        if until:
            segs = [s for s in segs if datetime.fromtimestamp(s[1], tz=timezone.utc) <= until]
        if not segs:
            return ""
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:6",
            "#EXT-X-PLAYLIST-TYPE:VOD",
            "#EXT-X-TARGETDURATION:7",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-INDEPENDENT-SEGMENTS",
        ]
        for p, _, dur in segs:
            lines.append(f"#EXTINF:{dur:.3f},")
            lines.append(p.name)
        lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines) + "\n"
