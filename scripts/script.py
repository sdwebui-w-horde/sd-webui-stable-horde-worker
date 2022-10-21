from typing import Optional

from fastapi import FastAPI
import gradio as gr

from modules import script_callbacks, shared

from stable_horde import StableHorde, StableHordeConfig

async def start_horde():
    config = StableHordeConfig()
    horde = StableHorde(config)
    await horde.run()


def on_app_started(demo: Optional[gr.Blocks], app: FastAPI):
    @app.get('/stable-horde')
    async def stable_horde():
        await start_horde()

    import requests
    if demo is None:
        local_url = f"http://localhost:{shared.cmd_opts.port if shared.cmd_opts.port else 7861}/"
    else:
        local_url = demo.local_url

    requests.get(f"{local_url}stable-horde")


def on_ui_settings():
    section = ('stable-horde', 'Stable Horde')
    shared.opts.add_option('stable_horde_enable', shared.OptionInfo(True, 'Enable', section=section))


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_settings(on_ui_settings)
