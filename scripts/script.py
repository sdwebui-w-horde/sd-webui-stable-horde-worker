from typing import Optional

from fastapi import FastAPI
import gradio as gr

from modules import scripts, script_callbacks, shared

from stable_horde import StableHorde, StableHordeConfig

basedir = scripts.basedir()
config = StableHordeConfig(basedir)
horde = StableHorde(basedir, config)

async def start_horde():
    await horde.run()


def on_app_started(demo: Optional[gr.Blocks], app: FastAPI):
    @app.get('/stable-horde')
    async def stable_horde():
        await start_horde()

    import requests
    if demo is None:
        local_url = f'http://localhost:{shared.cmd_opts.port if shared.cmd_opts.port else 7861}/'
    else:
        local_url = demo.local_url

    requests.get(f'{local_url}stable-horde')


def on_ui_tabs():
    tab_prefix = 'stable-horde-'
    with gr.Blocks() as demo:
        with gr.Row(elem_id='stable-horde').style(equal_height=False):
            with gr.Column():
                with gr.Row():
                    enable = gr.Button('Enable', visible=not config.enabled, variant='primary', elem_id=tab_prefix + 'enable', _js='stableHordeStartTimer')
                    disable = gr.Button('Disable', visible=config.enabled, elem_id=tab_prefix + 'disable', _js='stableHordeStopTimer')

                    def on_enable(enable: bool):
                        config.enabled = enable
                        return gr.update(visible=not enable), gr.update(visible=enable)
                    enable.click(fn=lambda: on_enable(True), inputs=[], outputs=[enable, disable], show_progress=False)
                    disable.click(fn=lambda: on_enable(False), inputs=[], outputs=[enable, disable],  show_progress=False)

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
                show_images = gr.Checkbox(config.show_image_preview, label='Show Images')

                refresh = gr.Button('Refresh', visible=False, elem_id=tab_prefix + 'refresh')
                refresh_image = gr.Button('Refresh Image', visible=False, elem_id=tab_prefix + 'refresh-image')

                current_id = gr.Textbox('Current ID: ', label='', elem_id=tab_prefix + 'current-id', readonly=True)
                preview = gr.Gallery(elem_id=tab_prefix + 'preview', visible=config.show_image_preview, readonly=True)

                def on_show_images(value: bool):
                    config.show_image_preview = value
                    return gr.update(visible=value)

                def on_refresh(image=False, show_images=config.show_image_preview):
                    cid = f"Current ID: {horde.state.id}"
                    html = ''.join(map(lambda x: f'<p>{x[0]}: {x[1]}</p>', horde.state.to_dict().items()))
                    images = [horde.state.image] if horde.state.image is not None else []
                    if image and show_images:
                        return cid, html, images
                    return cid, html

                show_images.change(fn=on_show_images, inputs=[show_images], outputs=[preview])
                with gr.Row():
                    log = gr.HTML(elem_id=tab_prefix + 'log')

                refresh.click(fn=lambda: on_refresh(), outputs=[current_id, log], show_progress=False)
                refresh_image.click(fn=lambda: on_refresh(True), outputs=[current_id, log, preview], show_progress=False)

    return (demo, 'Stable Horde', 'stable-horde'),


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
