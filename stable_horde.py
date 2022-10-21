import asyncio
import base64
import io
from random import randint
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from PIL import Image

class StableHordeConfig:
    def __init__(self):
        pass

    @property
    def endpoint(self) -> str:
        return shared.opts.stable_horde_endpoint
        
    @property
    def apikey(self) -> str:
        return shared.opts.stable_horde_apikey

    @property
    def name(self) -> str:
        return shared.opts.stable_horde_name

    @property
    def models(self) -> List[str]:
        return [shared.opts.stable_horde_model]

    @property
    def max_pixels(self) -> int:
        return int(shared.opts.stable_horde_max_pixels)

    @property
    def nsfw(self) -> bool:
        return shared.opts.stable_horde_nsfw

    @property
    def allow_img2img(self) -> bool:
        return shared.opts.stable_horde_allow_img2img

    @property
    def allow_painting(self) -> bool:
        return shared.opts.stable_horde_allow_painting

    @property
    def allow_unsafe_ipaddr(self) -> bool:
        return shared.opts.stable_horde_allow_unsafe_ipaddr


class StableHorde:
    def __init__(self, config: StableHordeConfig):
        self.config = config
        headers = {
            "apikey": self.config.apikey,
            "Content-Type": "application/json",
        }

        self.session = aiohttp.ClientSession(self.config.endpoint, headers=headers)

    async def run(self):
        while True:
            await asyncio.sleep(shared.opts.stable_horde_interval)

            if shared.opts.stable_horde_enable:
                try:
                    req = await self.get_popped_request()
                    if req is None:
                        continue

                    await self.handle_request(req)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
