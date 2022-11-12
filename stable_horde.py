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

    def patch_sampler_names(self):
        """Add more samplers that the Stable Horde supports,
        but are not included in the default sd_samplers module.
        """
        from modules import sd_samplers

        if sd_samplers.samplers_map.get('euler a karras'):
            # already patched
            return

        samplers = [
            sd_samplers.SamplerData("Euler a Karras", lambda model, funcname="sample_euler_ancestral": sd_samplers.KDiffusionSampler(funcname, model), ['k_euler_a_ka'], {'scheduler': 'karras'}),
            sd_samplers.SamplerData("Euler Karras", lambda model, funcname="sample_euler": sd_samplers.KDiffusionSampler(funcname, model), ['k_euler_ka'], {'scheduler': 'karras'}),
            sd_samplers.SamplerData("Heun Karras", lambda model, funcname="sample_heun": sd_samplers.KDiffusionSampler(funcname, model), ['k_heun_ka'], {'scheduler': 'karras'}),
            sd_samplers.SamplerData('DPM adaptive Karras', lambda model, funcname='sample_dpm_adaptive': sd_samplers.KDiffusionSampler(funcname, model), ['k_dpm_ad_ka'], {'scheduler': 'karras'}),
        ]
        sd_samplers.samplers.extend(samplers)
        sd_samplers.samplers_for_img2img.extend(samplers)
        sd_samplers.all_samplers_map.update({s.name: s for s in samplers})
        for sampler in samplers:
            sd_samplers.samplers_map[sampler.name.lower()] = sampler.name
            for alias in sampler.aliases:
                sd_samplers.samplers_map[alias.lower()] = sampler.name


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

        if r.status != 200:
            self.handle_error(r.status, req)
            return

        return req


    async def handle_request(self, req: Dict[str, Any]):
        if not req.get('id'):
            return

        self.patch_sampler_names()

        print(f"Get popped generation request {req['id']}: {req['payload'].get('prompt', '')}")
        sampler_name = req['payload']['sampler_name']
        if sampler_name == 'k_dpm_adaptive':
            sampler_name = 'k_dpm_ad'
        if sampler_name not in sd_samplers.samplers_map:
            print(f"ERROR: Unknown sampler {sampler_name}")
            return
        if req['payload']['karras']:
            sampler_name += '_ka'

        sampler = sd_samplers.samplers_map.get(sampler_name, None)
        if sampler is None:
            raise Exception(f"ERROR: Unknown sampler {sampler_name}")

        prompt = req['payload'].get('prompt', '')
        if "###" in prompt:
            prompt, negative = prompt.split("###")
        else:
            negative = ""

        postprocessors = req['payload'].get('post_processing', None) or []

        params = {
            "sd_model": shared.sd_model,
            "prompt": prompt.strip(),
            "negative_prompt": negative.strip(),
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
        image.save(bytesio, format="WebP", quality=95)

        if req.get("r2_upload"):
            async with aiohttp.ClientSession() as session:
                await session.put(req.get("r2_upload"), data=bytesio.getvalue())
            generation = "R2"

        else:
            generation = base64.b64encode(bytesio.getvalue()).decode("utf8")

        await self.submit(req['id'], req['payload']['seed'], generation)


    async def submit(self, id: str, seed: str, generation: str):
        post_data = {
            "id": id,
            "generation": generation,
            "seed": seed,
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
        elif (r.status == 400):
            print("ERROR: Generation Already Submitted")
        else:
            self.handle_error(r.status, res)

    def handle_error(self, status: int, res: Dict[str, Any]):
        if status == 401:
            print("ERROR: Invalid API Key")
        elif status == 403:
            print(f"ERROR: Access Denied. ({res.get('message', '')})")
        elif status == 404:
            print("ERROR: Request Not Found")
        else:
            print(f"ERROR: Unknown Error {status}")
            print(res)
