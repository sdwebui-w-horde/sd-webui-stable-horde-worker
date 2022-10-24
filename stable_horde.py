import asyncio
import base64
import io
from random import randint
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from PIL import Image

from modules import shared, call_queue, txt2img, processing, sd_samplers

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

    async def get_popped_request(self) -> Optional[Dict[str, Any]]:
        # https://stablehorde.net/api/
        post_data = {
            "name": self.config.name,
            "priority_usernames": [],
            "nsfw": self.config.nsfw,
            "blacklist": [],
            "models": self.config.models,
            "bridge_version": 8,
            "threads": 1,
            "max_pixels": self.config.max_pixels,
            "allow_img2img": self.config.allow_img2img,
            "allow_painting": self.config.allow_painting,
            "allow_unsafe_ipaddr": self.config.allow_unsafe_ipaddr,
        }

        r = await self.session.post('/api/v2/generate/pop', json=post_data)

        req = await r.json()

        if r.status == 200:
            return req


    async def handle_request(self, req: Dict[str, Any]):
        sampler_name = req['payload']['sampler_name']

        if req['payload']['karras']:
            sampler_name += '_ka'

        sampler = sd_samplers.samplers_map.get(sampler_name, None)

        params = {
            "sd_model": shared.sd_model,
            "prompt": req['payload']['prompt'],
            "sampler_name": sampler,
            "cfg_scale": req['payload'].get('cfg_scale', 5.0),
            "seed": req['payload'].get('seed', randint(0, 2**32)),
            "denoising_strength": req['payload'].get('denoising_strength', 0.75),
            "height": req['payload']['height'],
            "width": req['payload']['width'],
            "subseed": req['payload'].get('seed_variation', 1),
            "steps": req['payload']['ddim_steps'],
            "n_iter": req['payload']['n_iter'],
            "do_not_save_samples": True,
            "do_not_save_grid": True,
        }

        p = txt2img.StableDiffusionProcessingTxt2Img(**params)
        
        shared.state.begin()

        with call_queue.queue_lock:
            processed = processing.process_images(p)
            image = processed.images[0]

        shared.state.end()

        bytesio = io.BytesIO()
        image.save(bytesio, format="WebP", quality=75)

        generation = base64.b64encode(bytesio.getvalue()).decode("utf8")

        post_data = {
            "id": id,
            "generation": generation,
            "seed": params["seed"],
        }

        r = await self.session.post('/api/v2/generate/submit', json=post_data)

        res = await r.json()

        """
        res = {
            "reward": 10
        }
        """
        if (r.status == 200 and res.get("reward") is not None):
            print(f"Submission accepted, reward {res['reward']} received.")
