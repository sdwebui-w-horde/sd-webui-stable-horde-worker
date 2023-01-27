import asyncio
import base64
import io
import json
from os import path
from random import randint
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor

from modules.images import save_image
from modules import (
    shared,
    call_queue,
    txt2img,
    img2img,
    processing,
    sd_models,
    sd_samplers,
)

stable_horde_supported_models_url = (
    "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db.json"
)

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None


class StableHordeConfig(object):
    enabled: bool = False
    endpoint: str = "https://stablehorde.net/"
    apikey: str = "00000000"
    name: str = ""
    interval: int = 10
    max_pixels: int = 1048576  # 1024x1024
    nsfw: bool = False
    allow_img2img: bool = True
    allow_painting: bool = True
    allow_unsafe_ipaddr: bool = True
    allow_post_processing: bool = True
    show_image_preview: bool = False
    save_images: bool = False
    save_images_folder: str = "horde"

    def __init__(self, basedir: str):
        self.basedir = basedir
        self.config = self.load()

    def __getattribute__(self, item: str):
        if item in ["config", "basedir", "load", "save"]:
            return super().__getattribute__(item)
        value = self.config.get(item, None)
        if value is None:
            return super().__getattribute__(item)
        return value

    def __setattr__(self, key: str, value: Any):
        if key == "config" or key == "basedir":
            super().__setattr__(key, value)
        else:
            self.config[key] = value
            self.save()

    def load(self):
        if not path.exists(path.join(self.basedir, "config.json")):
            self.config = {
                "enabled": False,
                "allow_img2img": True,
                "allow_painting": True,
                "allow_unsafe_ipaddr": True,
                "allow_post_processing": True,
                "show_image_preview": False,
                "save_images": False,
                "save_images_folder": "horde",
                "endpoint": "https://stablehorde.net/",
                "apikey": "00000000",
                "name": "",
                "interval": 10,
                "max_pixels": 1048576,
                "nsfw": False,
            }
            self.save()

        with open(path.join(self.basedir, "config.json"), "r") as f:
            return json.load(f)

    def save(self):
        with open(path.join(self.basedir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)


class State:
    def __init__(self):
        self._status = ""
        self.id: Optional[str] = None
        self.prompt: Optional[str] = None
        self.negative_prompt: Optional[str] = None
        self.scale: Optional[float] = None
        self.steps: Optional[int] = None
        self.sampler: Optional[str] = None
        self.image: Optional[Image.Image] = None

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        if shared.cmd_opts.nowebui:
            print(value)

    def to_dict(self):
        return {
            "status": self.status,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "scale": self.scale,
            "steps": self.steps,
            "sampler": self.sampler,
        }


class HordeJob:
    retry_interval: int = 1

    def __init__(
        self,
        session: aiohttp.ClientSession,
        id: str,
        model: str,
        prompt: str,
        negative_prompt: str,
        sampler: str,
        cfg_scale: float,
        seed: int,
        denoising_strength: float,
        n_iter: int,
        height: int,
        width: int,
        subseed: int,
        steps: int,
        karras: bool,
        tiling: bool,
        postprocessors: List[str],
        nsfw_censor: bool = False,
        source_image: Optional[Image.Image] = None,
        source_processing: Optional[str] = "img2img",
        source_mask: Optional[Image.Image] = None,
        r2_upload: Optional[str] = None,
    ):
        self.id = id
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.sampler = sampler
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.denoising_strength = denoising_strength
        self.n_iter = n_iter
        self.height = height
        self.width = width
        self.subseed = subseed
        self.steps = steps
        self.karras = karras
        # TODO: add support for bridge version 11 "tiling"
        self.tiling = tiling
        self.postprocessors = postprocessors
        self.nsfw_censor = nsfw_censor
        self.source_image = source_image
        self.source_processing = (
            source_processing  # "img2img", "inpainting", "outpainting"
        )
        self.source_mask = source_mask
        self.r2_upload = r2_upload

    async def submit(self, image: Image.Image, session: aiohttp.ClientSession):
        bytesio = io.BytesIO()
        image.save(bytesio, format="WebP", quality=95)

        if self.r2_upload:
            async with aiohttp.ClientSession() as session:
                attempts = 10
                while attempts > 0:
                    try:
                        r = await session.put(self.r2_upload, data=bytesio.getvalue())
                        break
                    except aiohttp.ClientConnectorError:
                        attempts -= 1
                        await asyncio.sleep(self.retry_interval)
                        continue
            generation = "R2"

        else:
            generation = base64.b64encode(bytesio.getvalue()).decode("utf8")

        post_data = {
            "id": self.id,
            "generation": generation,
            "seed": self.seed,
        }

        attempts = 10
        while attempts > 0:
            try:
                r = await session.post("/api/v2/generate/submit", json=post_data)

                try:
                    res = await r.json()

                    if r.status == 404:
                        print(f"job {self.id} has been submitted already")
                        return

                    if r.status == 500:
                        print(
                            f"Failed to submit job with status code {r.status}, retry!"
                        )
                        attempts -= 1
                        await asyncio.sleep(self.retry_interval)
                        continue

                    if r.ok:
                        return res.get("reward", None)
                    else:
                        print(
                            "Failed to submit job with status code"
                            + f"{r.status}: {res.get('message')}"
                        )
                        return None
                except Exception:
                    print("Error when decoding response, the server might be down.")
                    return None

            except aiohttp.ClientConnectorError:
                attempts -= 1
                await asyncio.sleep(self.retry_interval)
                continue

    @classmethod
    async def get(
        cls,
        session: aiohttp.ClientSession,
        config: StableHordeConfig,
        models: List[str],
    ):
        name = "Stable Horde Worker Bridge for Stable Diffusion WebUI"
        version = 10
        repo = "https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker"
        # https://stablehorde.net/api/
        post_data = {
            "name": config.name,
            "priority_usernames": [],
            "nsfw": config.nsfw,
            "blacklist": [],
            "models": models,
            # TODO: add support for bridge version 11 "tiling"
            "bridge_version": 9,
            "bridge_agent": f"{name}:{version}:{repo}",
            "threads": 1,
            "max_pixels": config.max_pixels,
            "allow_img2img": config.allow_img2img,
            "allow_painting": config.allow_painting,
            "allow_unsafe_ipaddr": config.allow_unsafe_ipaddr,
        }

        r = await session.post("/api/v2/generate/pop", json=post_data)

        req = await r.json()

        if r.status != 200:
            raise Exception(f"Failed to get job: {req.get('message')}")

        if not req.get("id"):
            return

        payload = req.get("payload")
        prompt = payload.get("prompt")
        if "###" in prompt:
            prompt, negative = map(lambda x: x.strip(), prompt.rsplit("###", 1))
        else:
            negative = ""

        def to_image(base64str: Optional[str]) -> Optional[Image.Image]:
            if not base64str:
                return None
            return Image.open(io.BytesIO(base64.b64decode(base64str)))

        return cls(
            session=session,
            id=req["id"],
            prompt=prompt,
            negative_prompt=negative,
            sampler=payload.get("sampler_name"),
            cfg_scale=payload.get("cfg_scale", 5),
            seed=int(payload.get("seed", randint(0, 2**32))),
            denoising_strength=payload.get("denoising_strength", 0.75),
            n_iter=payload.get("n_iter", 1),
            height=payload["height"],
            width=payload["width"],
            subseed=payload.get("seed_variation", 1),
            steps=payload.get("ddim_steps", 30),
            karras=payload.get("karras", False),
            tiling=payload.get("tiling", False),
            postprocessors=payload.get("post_processing", []),
            nsfw_censor=payload.get("use_nsfw_censor", False),
            model=req["model"],
            source_image=to_image(payload.get("source_image")),
            source_processing=payload.get("source_processing"),
            source_mask=to_image(payload.get("source_mask")),
            r2_upload=payload.get("r2_upload"),
        )


class StableHorde:
    def __init__(self, basedir: str, config: StableHordeConfig):
        self.basedir = basedir
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

        self.sfw_request_censor = Image.open(
            path.join(self.config.basedir, "assets", "nsfw_censor_sfw_request.png")
        )

        self.supported_models = []
        self.current_models = []

        self.state = State()

    async def get_supported_models(self):
        filepath = path.join(self.basedir, "stablehorde_supported_models.json")
        if not path.exists(filepath):
            async with aiohttp.ClientSession() as session:
                async with session.get(stable_horde_supported_models_url) as resp:
                    with open(filepath, "wb") as f:
                        f.write(await resp.read())
        with open(filepath, "r") as f:
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
            return f"Model checkpoint {model_checkpoint} not found"

        local_hash = get_md5sum(checkpoint_info.filename)
        for model in self.supported_models:
            try:
                remote_hash = model["config"]["files"][0]["md5sum"]
            except KeyError:
                continue

            if local_hash == remote_hash:
                self.current_models = [model["name"]]

        if len(self.current_models) == 0:
            return f"Current model {model_checkpoint} not found on StableHorde"

    async def run(self):
        await self.get_supported_models()

        while True:
            result = self.detect_current_model()
            if result is not None:
                self.state.status = result
                # Wait 10 seconds before retrying to detect the current model
                # if the current model is not listed in the Stable Horde supported
                # models, we don't want to spam the server with requests
                await asyncio.sleep(10)
                continue

            await asyncio.sleep(self.config.interval)

            if self.config.enabled:
                try:
                    req = await HordeJob.get(
                        await self.get_session(), self.config, self.current_models
                    )
                    if req is None:
                        continue

                    await self.handle_request(req)
                except Exception:
                    import traceback

                    traceback.print_exc()

    def patch_sampler_names(self):
        """Add more samplers that the Stable Horde supports,
        but are not included in the default sd_samplers module.
        """
        from modules import sd_samplers
        from modules.sd_samplers import KDiffusionSampler, SamplerData

        if sd_samplers.samplers_map.get("euler a karras"):
            # already patched
            return

        samplers = [
            SamplerData(
                "Euler a Karras",
                lambda model, funcname="sample_euler_ancestral": KDiffusionSampler(
                    funcname, model
                ),
                ["k_euler_a_ka"],
                {"scheduler": "karras"},
            ),
            SamplerData(
                "Euler Karras",
                lambda model, funcname="sample_euler": KDiffusionSampler(
                    funcname, model
                ),
                ["k_euler_ka"],
                {"scheduler": "karras"},
            ),
            SamplerData(
                "Heun Karras",
                lambda model, funcname="sample_heun": KDiffusionSampler(
                    funcname, model
                ),
                ["k_heun_ka"],
                {"scheduler": "karras"},
            ),
            SamplerData(
                "DPM adaptive Karras",
                lambda model, funcname="sample_dpm_adaptive": KDiffusionSampler(
                    funcname, model
                ),
                ["k_dpm_ad_ka"],
                {"scheduler": "karras"},
            ),
            SamplerData(
                "DPM fast Karras",
                lambda model, funcname="sample_dpm_fast": KDiffusionSampler(
                    funcname, model
                ),
                ["k_dpm_fast_ka"],
                {"scheduler": "karras"},
            ),
        ]
        sd_samplers.samplers.extend(samplers)
        sd_samplers.samplers_for_img2img.extend(samplers)
        sd_samplers.all_samplers_map.update({s.name: s for s in samplers})
        for sampler in samplers:
            sd_samplers.samplers_map[sampler.name.lower()] = sampler.name
            for alias in sampler.aliases:
                sd_samplers.samplers_map[alias.lower()] = sampler.name

    async def handle_request(self, job: HordeJob):
        self.patch_sampler_names()

        self.state.status = f"Get popped generation request {job.id}"
        sampler_name = job.sampler
        if sampler_name == "k_dpm_adaptive":
            sampler_name = "k_dpm_ad"
        if sampler_name not in sd_samplers.samplers_map:
            self.state.status = f"ERROR: Unknown sampler {sampler_name}"
            return
        if job.karras:
            sampler_name += "_ka"

        sampler = sd_samplers.samplers_map.get(sampler_name, None)
        if sampler is None:
            raise Exception(f"ERROR: Unknown sampler {sampler_name}")

        postprocessors = job.postprocessors

        params = {
            "sd_model": shared.sd_model,
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt,
            "sampler_name": sampler,
            "cfg_scale": job.cfg_scale,
            "seed": job.seed,
            "denoising_strength": job.denoising_strength,
            "height": job.height,
            "width": job.width,
            "subseed": job.subseed,
            "steps": job.steps,
            "n_iter": job.n_iter,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
        }

        if job.source_image is not None:
            p = img2img.StableDiffusionProcessingImg2Img(
                init_images=[job.source_image],
                mask=job.source_mask,
                **params,
            )
        else:
            p = txt2img.StableDiffusionProcessingTxt2Img(**params)

        with call_queue.queue_lock:
            shared.state.begin()
            processed = processing.process_images(p)
            shared.state.end()

        has_nsfw = False

        with call_queue.queue_lock:
            if job.nsfw_censor:
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
                with call_queue.queue_lock:
                    image = face_restorators[0].restore(np.array(image))
                image = Image.fromarray(image)

        if "RealESRGAN_x4plus" in postprocessors and not has_nsfw:
            from modules.postprocessing import run_extras

            with call_queue.queue_lock:
                images, _info, _wtf = run_extras(
                    image=image,
                    extras_mode=0,
                    resize_mode=0,
                    show_extras_results=True,
                    upscaling_resize=2,
                    upscaling_resize_h=None,
                    upscaling_resize_w=None,
                    upscaling_crop=False,
                    upscale_first=False,
                    extras_upscaler_1="R-ESRGAN 4x+",  # 8 - RealESRGAN_x4plus
                    extras_upscaler_2=None,
                    extras_upscaler_2_visibility=0.0,
                    gfpgan_visibility=0.0,
                    codeformer_visibility=0.0,
                    codeformer_weight=0.0,
                    image_folder="",
                    input_dir="",
                    output_dir="",
                    save_output=False,
                )

            image = images[0]

        # Saving image locally
        infotext = (
            processing.create_infotext(
                p, p.all_prompts, p.all_seeds, p.all_subseeds, "Stable Horde", 0, 0
            )
            if shared.opts.enable_pnginfo
            else None
        )
        if self.config.save_images:
            save_image(
                image,
                self.config.save_images_folder,
                "",
                job.seed,
                job.prompt,
                "png",
                info=infotext,
                p=p,
            )

        self.state.id = job.id
        self.state.prompt = job.prompt
        self.state.negative_prompt = job.negative_prompt
        self.state.scale = job.cfg_scale
        self.state.steps = job.steps
        self.state.sampler = sampler_name
        self.state.image = image

        res = await job.submit(image, await self.get_session())
        if res:
            self.state.status = f"Submission accepted, reward {res} received."

    # check and replace nsfw content
    def check_safety(self, x_image):
        global safety_feature_extractor, safety_checker

        if safety_feature_extractor is None:
            safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
                safety_model_id
            )
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                safety_model_id
            )

        safety_checker_input = safety_feature_extractor(x_image, return_tensors="pt")
        image, has_nsfw_concept = safety_checker(
            images=x_image, clip_input=safety_checker_input.pixel_values
        )

        if has_nsfw_concept:
            return self.sfw_request_censor, has_nsfw_concept
        return Image.fromarray(image), has_nsfw_concept

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            headers = {
                "apikey": self.config.apikey,
                "Content-Type": "application/json",
            }
            self.session = aiohttp.ClientSession(self.config.endpoint, headers=headers)
        return self.session

    def handle_error(self, status: int, res: Dict[str, Any]):
        if status == 401:
            self.state.status = "ERROR: Invalid API Key"
        elif status == 403:
            self.state.status = f"ERROR: Access Denied. ({res.get('message', '')})"
        elif status == 404:
            self.state.status = "ERROR: Request Not Found"
        else:
            self.state.status = f"ERROR: Unknown Error {status}"
            print(f"ERROR: Unknown Error, {res}")
