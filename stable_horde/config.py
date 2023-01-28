import json
import os.path as path
from typing import Any


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
