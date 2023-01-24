import asyncio
import base64
import io
import json
from os import path
from random import randint
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from transformers import AutoFeatureExtractor

from modules import shared, call_queue, txt2img, img2img, processing, sd_models, sd_samplers, scripts

stable_horde_supported_models_url = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db.json"

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None


class StableHordeConfig:
    def __init__(self, basedir: str):
        self.basedir = basedir
        self.models = []

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

        self.sfw_request_censor = Image.open(path.join(self.config.basedir, "assets", "nsfw_censor_sfw_request.png"))

        self.supported_models = []

    async def get_supported_models(self):
        filepath = path.join(self.config.basedir, "stablehorde_supported_models.json")
        if not path.exists(filepath):
            async with aiohttp.ClientSession() as session:
                async with session.get(stable_horde_supported_models_url) as resp:
                    with open(filepath, 'wb') as f:
                        f.write(await resp.read())
        with open(filepath, 'r') as f:
            supported_models: Dict[str, Any] = json.load(f)

        self.supported_models = list(supported_models.values())

    def detect_current_model(self):
        def get_md5sum(filepath):
            import hashlib
            with open(filepath, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()

        model_checkpoint = shared.opts.sd_model_checkpoint
        checkpoint_info = sd_models.checkpoints_list.get(model_checkpoint, None)
        if checkpoint_info is None:
            raise Exception(f"Model checkpoint {model_checkpoint} not found")

        local_hash = get_md5sum(checkpoint_info.filename)
        for model in self.supported_models:
            try:
                remote_hash = model["config"]["files"][0]["md5sum"]
            except KeyError:
                continue

            if local_hash == remote_hash:
                self.config.models = [model["name"]]

        if len(self.config.models) == 0:
            raise Exception(f"Current model {model_checkpoint} not found on StableHorde")


    async def run(self):
        await self.get_supported_models()
        self.detect_current_model()

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
            # TODO: add support for bridge version 11 "tiling"
            "bridge_version": 9,
            "bridge_agent": "Stable Horde Worker Bridge for Stable Diffusion WebUI:10:https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker",
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

        if req.get('source_image', None) is not None:
            b64 = req.get('source_image')
            image = Image.open(io.BytesIO(base64.b64decode(b64)))
            mask = None
            if req.get('source_mask', None) is not None:
                b64 = req.get('source_mask')
                mask = Image.open(io.BytesIO(base64.b64decode(b64)))
            p = img2img.StableDiffusionProcessingImg2Img(
            init_images=[image],
            mask=mask,
            **params,
        )
        else:
            p = txt2img.StableDiffusionProcessingTxt2Img(**params)
        
        shared.state.begin()

        with call_queue.queue_lock:
            processed = processing.process_images(p)

            has_nsfw = False

            if req["payload"].get("use_nsfw_censor"):
                x_image = np.array(processed.images[0])
                image, has_nsfw = self.check_safety(x_image)

            else:
                image = processed.images[0]

            if "GFPGAN" in postprocessors or "CodeFormers" in postprocessors:
                model = "CodeFormer" if "CodeFormers" in postprocessors else "GFPGAN"
                face_restorators = [x for x in shared.face_restorers if x.name() == model]
                if len(face_restorators) == 0:
                    print(f"ERROR: No face restorer for {model}")

                else:
                    image = face_restorators[0].restore(np.array(image))
                    image = Image.fromarray(image)

            if "RealESRGAN_x4plus" in postprocessors and not has_nsfw:
                from modules.postprocessing import run_extras
                images, _info, _wtf = run_extras(
                    image=image, extras_mode=0, resize_mode=0,
                    show_extras_results=True, upscaling_resize=2,
                    upscaling_resize_h=None, upscaling_resize_w=None,
                    upscaling_crop=False, upscale_first=False,
                    extras_upscaler_1="R-ESRGAN 4x+", # 8 - RealESRGAN_x4plus
                    extras_upscaler_2=None,
                    extras_upscaler_2_visibility=0.0,
                    gfpgan_visibility=0.0, codeformer_visibility=0.0, codeformer_weight=0.0,
                    image_folder="", input_dir="", output_dir="",
                )

                image = images[0]

        shared.state.end()

        bytesio = io.BytesIO()
        image.save(bytesio, format="WebP", quality=95)

        if req.get("r2_upload"):
            async with aiohttp.ClientSession() as session:
                await session.put(req.get("r2_upload"), data=bytesio.getvalue())
            generation = "R2"

        else:
            generation = base64.b64encode(bytesio.getvalue()).decode("utf8")

        await self.submit(req['id'], int(req['payload']['seed']), generation)


    async def submit(self, id: str, seed: int, generation: str):
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


    # check and replace nsfw content
    def check_safety(self, x_image):
        global safety_feature_extractor, safety_checker

        if safety_feature_extractor is None:
            safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

        safety_checker_input = safety_feature_extractor(x_image, return_tensors="pt")
        image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)

        if has_nsfw_concept:
            return self.sfw_request_censor, has_nsfw_concept
        return Image.fromarray(image), has_nsfw_concept


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
