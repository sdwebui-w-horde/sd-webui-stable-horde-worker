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


def apply_stable_horde_settings(enable: bool, name: str, apikey: str, allow_img2img: bool, allow_painting: bool, allow_unsafe_ipaddr: bool, allow_post_processing, nsfw: bool, interval: int, max_pixels: str, endpoint: str, model_selection: list, show_images: bool):
    config.enabled = enable
    config.allow_img2img = allow_img2img
    config.allow_painting = allow_painting
    config.allow_unsafe_ipaddr = allow_unsafe_ipaddr
    config.allow_post_processing = allow_post_processing
    config.interval = interval
    config.endpoint = endpoint
    config.apikey = apikey
    config.name = name
    config.max_pixels = int(max_pixels)
    config.nsfw = nsfw
    config.show_image_preview = show_images
    config.save()

    return f'Status: {"Running" if config.enabled else "Stopped"}', 'Running Type: Image Generation'


def on_ui_tabs():
    tab_prefix = 'stable-horde-'
    with gr.Blocks() as demo:
        with gr.Column(elem_id='stable-horde'):
            with gr.Row():
                status = gr.Textbox(f'Status: {"Running" if config.enabled else "Stopped"}', label='', elem_id=tab_prefix + 'status', readonly=True)
                running_type = gr.Textbox('Running Type: Image Generation', label='', elem_id=tab_prefix + 'running-type', readonly=True)

                apply_settings = gr.Button('Apply Settings', elem_id=tab_prefix + 'apply-settings')
            with gr.Row():
                state = gr.Textbox('', label='', readonly=True)
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Box(scale=2):
                        enable = gr.Checkbox(config.enabled, label='Enable', elem_id=tab_prefix + 'enable')
                        name = gr.Textbox(config.name, label='Worker Name', elem_id=tab_prefix + 'name')
                        apikey = gr.Textbox(config.apikey, label='Stable Horde API Key', elem_id=tab_prefix + 'apikey')
                        allow_img2img = gr.Checkbox(config.allow_img2img, label='Allow img2img')
                        allow_painting = gr.Checkbox(config.allow_painting, label='Allow Painting')
                        allow_unsafe_ipaddr = gr.Checkbox(config.allow_unsafe_ipaddr, label='Allow Unsafe IP Address')
                        allow_post_processing = gr.Checkbox(config.allow_post_processing, label='Allow Post Processing')
                        nsfw = gr.Checkbox(config.nsfw, label='Allow NSFW')
                        interval = gr.Slider(0, 60, config.interval, step=1, label='Duration Between Generations (seconds)')
                        max_pixels = gr.Textbox(str(config.max_pixels), label='Max Pixels', elem_id=tab_prefix + 'max-pixels')
                        endpoint = gr.Textbox(config.endpoint, label='Stable Horde Endpoint', elem_id=tab_prefix + 'endpoint')

                    with gr.Row():
                        model_selection = gr.CheckboxGroup(choices=shared.list_checkpoint_tiles(), value=[shared.list_checkpoint_tiles()[0]], label='Model Selection')

                with gr.Column():
                    show_images = gr.Checkbox(config.show_image_preview, label='Show Images')

                    refresh = gr.Button('Refresh', visible=False, elem_id=tab_prefix + 'refresh')
                    refresh_image = gr.Button('Refresh Image', visible=False, elem_id=tab_prefix + 'refresh-image')

                    current_id = gr.Textbox('Current ID: ', label='', elem_id=tab_prefix + 'current-id', readonly=True)
                    preview = gr.Gallery(label='Preview', elem_id=tab_prefix + 'preview', visible=config.show_image_preview, readonly=True).style(grid=4)

                    def on_refresh(image=False, show_images=config.show_image_preview):
                        cid = f"Current ID: {horde.state.id}"
                        html = ''.join(map(lambda x: f'<p>{x[0]}: {x[1]}</p>', horde.state.to_dict().items()))
                        images = [horde.state.image] if horde.state.image is not None else []
                        if image and show_images:
                            return cid, html, horde.state.status, images
                        return cid, html, horde.state.status

                    with gr.Row():
                        log = gr.HTML(elem_id=tab_prefix + 'log')

                    refresh.click(fn=lambda: on_refresh(), outputs=[current_id, log, state], show_progress=False)
                    refresh_image.click(fn=lambda: on_refresh(True), outputs=[current_id, log, state, preview], show_progress=False)
        apply_settings.click(
            fn=apply_stable_horde_settings,
            inputs=[
                enable,
                name,
                apikey,
                allow_img2img,
                allow_painting,
                allow_unsafe_ipaddr,
                allow_post_processing,
                nsfw,
                interval,
                max_pixels,
                endpoint,
                model_selection,
                show_images,
            ],
            outputs=[status, running_type],
        )

    return (demo, 'Stable Horde Worker', 'stable-horde'),


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
