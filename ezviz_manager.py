"""EZVIZ Open Platform token manager.

Handles AccessToken acquisition and caching for EZUIKit player integration.
"""
import time
import logging
import requests

logger = logging.getLogger(__name__)

EZVIZ_APP_KEY = "6f344f5abbcd400689c197e93fdd1a42"
EZVIZ_APP_SECRET = "e05c2df6776345df8fe4e3b3152cd8d2"
EZVIZ_API_BASE = "https://isgpopen.ezvizlife.com"
TOKEN_REFRESH_MARGIN = 3600  # refresh 1 hour before expiry


class EzvizTokenManager:
    def __init__(self):
        self._token: str = ""
        self._expire_time: float = 0  # epoch ms

    def get_token(self) -> dict | None:
        """Return cached token or fetch a new one."""
        now_ms = time.time() * 1000
        if self._token and self._expire_time > now_ms + TOKEN_REFRESH_MARGIN * 1000:
            return {
                "accessToken": self._token,
                "expireTime": self._expire_time,
                "appKey": EZVIZ_APP_KEY,
            }
        return self._fetch_token()

    def _fetch_token(self) -> dict | None:
        try:
            resp = requests.post(
                f"{EZVIZ_API_BASE}/api/lapp/token/get",
                data={
                    "appKey": EZVIZ_APP_KEY,
                    "appSecret": EZVIZ_APP_SECRET,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("code") == "200":
                token_data = data["data"]
                self._token = token_data["accessToken"]
                self._expire_time = token_data["expireTime"]
                logger.info("EZVIZ token refreshed, expires at %s", self._expire_time)
                return {
                    "accessToken": self._token,
                    "expireTime": self._expire_time,
                    "appKey": EZVIZ_APP_KEY,
                }
            else:
                logger.error("EZVIZ token error: %s", data)
                return None
        except Exception as e:
            logger.error("Failed to fetch EZVIZ token: %s", e)
            return None
