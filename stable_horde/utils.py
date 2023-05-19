from typing import Any, Dict, Optional

from requests import Session
from urllib.parse import urljoin


class HordeRequestSession(Session):
    def __init__(self, base_url: str, base_headers: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.base_url = base_url
        self.base_headers = base_headers or {}

    def _set_base_url(self, base_url: str):
        self.base_url = base_url

    def request(self, method, url, *args, **kwargs):
        joined_url = urljoin(self.base_url, url)
        kwargs.setdefault("headers", {}).update(self.base_headers)
        return super().request(method, joined_url, *args, **kwargs)
