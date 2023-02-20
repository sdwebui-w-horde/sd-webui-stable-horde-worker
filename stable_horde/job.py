import asyncio
import base64
from enum import Enum
import io
from random import randint
from typing import List, Optional
from PIL import Image

import aiohttp
from .config import StableHordeConfig


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    GENERATED = "generated"
    SUBMITTING = "submitting"
    UPLOADED = "uploaded"
    SUBMITTED = "submitted"
    DONE = "done"
    ERROR = "error"


class HordeJob:
    retry_interval: int = 1
    censored = False

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
        self.status: JobStatus = JobStatus.PENDING
        self.session = session
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
        self.tiling = tiling
        self.postprocessors = postprocessors
        self.nsfw_censor = nsfw_censor
        self.source_image = source_image
        self.source_processing = (
            source_processing  # "img2img", "inpainting", "outpainting"
        )
        self.source_mask = source_mask
        self.r2_upload = r2_upload

    async def submit(self, image: Image.Image):
        self.status = JobStatus.SUBMITTING

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

            self.status = JobStatus.UPLOADED

        else:
            generation = base64.b64encode(bytesio.getvalue()).decode("utf8")

        post_data = {
            "id": self.id,
            "generation": generation,
            "seed": self.seed,
            "state": "censored" if self.censored else "ok",
        }

        attempts = 10
        while attempts > 0:
            try:
                r = await self.session.post("/api/v2/generate/submit", json=post_data)

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
                        self.status = JobStatus.SUBMITTED
                        reward = res.get("reward", None)
                        if reward:
                            self.status = JobStatus.DONE
                            return reward
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

        self.status = JobStatus.ERROR

    async def error(self):
        self.status = JobStatus.ERROR

        post_data = {"id": self.id, "state": "faulted"}
        attempts = 10
        while attempts > 0:
            try:
                r = await self.session.post("/api/v2/generate/submit", json=post_data)
                if r.ok:
                    print("Successfully reported error to Stable Horde")
                    return
                else:
                    res = await r.json()
                    print(
                        "Failed to report error with status code"
                        + f"{r.status}: {res.get('message')}"
                    )
                    return
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
        # Stable Horde uses a bridge version to differentiate between different
        # bridge agents which is used to determine the bridge agent's capabilities.
        # We should increment the version number when we add new features to the bridge
        # agent.
        #
        # When we increment the version number, we should also update the AI-Horde side:
        # https://github.com/db0/AI-Horde/blob/main/horde/bridge_reference.py
        #
        # 1 - img2img, inpainting, karras, r2, CodeFormers
        # 2 - tiling
        version = 2
        name = "SD-WebUI Stable Horde Worker Bridge"
        repo = "https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker"
        # https://stablehorde.net/api/
        post_data = {
            "name": config.name,
            "priority_usernames": [],
            "nsfw": config.nsfw,
            "blacklist": [],
            "models": models,
            # TODO: add support for bridge version 13 (clip_skip, hires_fix)
            # and 14 (r2_source)
            "bridge_version": 11,
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
            source_image=to_image(req.get("source_image")),
            source_processing=req.get("source_processing"),
            source_mask=to_image(req.get("source_mask")),
            r2_upload=req.get("r2_upload"),
        )
