from typing import Optional

from fastapi import FastAPI
import gradio as gr

from modules import scripts, script_callbacks, shared

from stable_horde import StableHorde, StableHordeConfig

basedir = scripts.basedir()


def on_app_started(demo: Optional[gr.Blocks], app: FastAPI):
    config = StableHordeConfig(basedir)
    horde = StableHorde(config)

    import gradio.utils
    gradio.utils.synchronize_async(horde.run)


def on_ui_settings():
    section = ('stable-horde', 'Stable Horde')
    shared.opts.add_option('stable_horde_enable', shared.OptionInfo(False, 'Enable', section=section))
    shared.opts.add_option('stable_horde_endpoint', shared.OptionInfo('https://stablehorde.net/', 'Endpoint', section=section))
    shared.opts.add_option('stable_horde_apikey', shared.OptionInfo('', 'API Key', section=section))
    shared.opts.add_option('stable_horde_name', shared.OptionInfo('Stable Horde', 'Worker Name', section=section))
    shared.opts.add_option('stable_horde_model', shared.OptionInfo('Anything Diffusion', 'Model', section=section))
    shared.opts.add_option('stable_horde_nsfw', shared.OptionInfo(False, 'NSFW', section=section))
    shared.opts.add_option('stable_horde_interval', shared.OptionInfo(10, 'Interval', section=section))
    shared.opts.add_option('stable_horde_max_pixels', shared.OptionInfo(1024 * 1024, 'Max Pixels', section=section))
    shared.opts.add_option('stable_horde_allow_img2img', shared.OptionInfo(True, 'Allow img2img', section=section))
    shared.opts.add_option('stable_horde_allow_painting', shared.OptionInfo(True, 'Allow Painting', section=section))
    shared.opts.add_option('stable_horde_allow_unsafe_ipaddr', shared.OptionInfo(True, 'Allow Unsafe IP Address', section=section))


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_settings(on_ui_settings)
