from typing import Optional

from fastapi import FastAPI
import gradio as gr

from modules import scripts, script_callbacks, shared

from stable_horde import StableHorde, StableHordeConfig

basedir = scripts.basedir()
config = StableHordeConfig(basedir)

async def start_horde(config: StableHordeConfig):
    horde = StableHorde(config)
    await horde.run()


def on_app_started(demo: Optional[gr.Blocks], app: FastAPI):
    @app.get('/stable-horde')
    async def stable_horde():
        await start_horde(config)

    import requests
    if demo is None:
        local_url = f'http://localhost:{shared.cmd_opts.port if shared.cmd_opts.port else 7861}/'
    else:
        local_url = demo.local_url

    requests.get(f'{local_url}stable-horde')


def on_ui_settings():
    section = ('stable-horde', 'Stable Horde')
    shared.opts.add_option('stable_horde_endpoint', shared.OptionInfo('https://stablehorde.net/', 'Endpoint', section=section))
    shared.opts.add_option('stable_horde_apikey', shared.OptionInfo('', 'API Key', section=section))
    shared.opts.add_option('stable_horde_name', shared.OptionInfo('Stable Horde', 'Team Name', section=section))
    shared.opts.add_option('stable_horde_nsfw', shared.OptionInfo(False, 'NSFW', section=section))
    shared.opts.add_option('stable_horde_interval', shared.OptionInfo(10, 'Interval', section=section))
    shared.opts.add_option('stable_horde_max_pixels', shared.OptionInfo(1024 * 1024, 'Max Pixels', section=section))


def on_ui_tabs():
    tabprefix = 'stable-horde-'
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    def enable_text():
                        return 'Disable' if config.enabled else 'Enable'

                    button = gr.Button(enable_text(), variant='primary', elem_id=tabprefix + 'enable')

                    def on_enable():
                        config.enabled = not config.enabled
                        button.label = enable_text()
                    button.click(fn=on_enable)

                    maintenance_mode = gr.Checkbox(config.maintenance, label='Maintenance Mode')
                    maintenance_mode.change(fn=lambda value: config.__setattr__("maintenance", value))

                with gr.Column(scale=2):
                    allow_img2img = gr.Checkbox(True, label='Allow img2img')
                    allow_img2img.change(fn=lambda value: config.__setattr__("allow_img2img", value))

                    allow_painting = gr.Checkbox(True, label='Allow Painting')
                    allow_painting.change(fn=lambda value: config.__setattr__("allow_painting", value))

                    allow_unsafe_ipaddr = gr.Checkbox(True, label='Allow Unsafe IP Address')
                    allow_unsafe_ipaddr.change(fn=lambda value: config.__setattr__("allow_unsafe_ipaddr", value))

                with gr.Row():
                    model_selection = gr.CheckboxGroup(choices=shared.list_checkpoint_tiles(), value=[shared.list_checkpoint_tiles()[0]], label='Model Selection')

            with gr.Column():
                show_images = gr.Checkbox(False, label='Show Images')
                preview = gr.Image(elem_id=tabprefix + 'preview', visible=False)
                with gr.Row():
                    log = gr.Textbox(elem_id=tabprefix + 'log', lines=10, width=400, height=400, readonly=True)

    return (demo, 'Stable Horde', 'stable-horde'),


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
