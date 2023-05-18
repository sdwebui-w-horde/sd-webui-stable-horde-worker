from typing import Optional
import time

from fastapi import FastAPI
import gradio as gr

from modules import scripts, script_callbacks, sd_models

from stable_horde import StableHorde, StableHordeConfig

basedir = scripts.basedir()
config = StableHordeConfig(basedir)
horde = StableHorde(basedir, config)


def on_app_started(demo: Optional[gr.Blocks], app: FastAPI):
    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_until_complete(horde.run())


def apply_stable_horde_settings(
    enable: bool,
    name: str,
    allow_img2img: bool,
    allow_painting: bool,
    allow_unsafe_ipaddr: bool,
    allow_post_processing,
    restore_settings: bool,
    nsfw: bool,
    interval: int,
    max_pixels: str,
    endpoint: str,
    show_images: bool,
    save_images: bool,
    save_images_folder: str,
):
    config.enabled = enable
    config.allow_img2img = allow_img2img
    config.allow_painting = allow_painting
    config.allow_unsafe_ipaddr = allow_unsafe_ipaddr
    config.allow_post_processing = allow_post_processing
    config.restore_settings = restore_settings
    config.interval = interval
    config.endpoint = endpoint
    config.name = name
    config.max_pixels = int(max_pixels)
    config.nsfw = nsfw
    config.show_image_preview = show_images
    config.save_images = save_images
    config.save_images_folder = save_images_folder
    config.save()

    return (
        f'Status: {"Running" if config.enabled else "Stopped"}',
        "Running Type: Image Generation",
    )


tab_prefix = "stable-horde-"


def get_worker_ui():
    with gr.Blocks() as worker_ui:
        with gr.Column(elem_id="stable-horde"):
            with gr.Row():
                status = gr.Textbox(
                    f'Status: {"Running" if config.enabled else "Stopped"}',
                    label="",
                    elem_id=tab_prefix + "status",
                    readonly=True,
                )
                running_type = gr.Textbox(
                    "Running Type: Image Generation",
                    label="",
                    elem_id=tab_prefix + "running-type",
                    readonly=True,
                )

                apply_settings = gr.Button(
                    "Apply Settings", elem_id=tab_prefix + "apply-settings"
                )
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Box(scale=2):
                        enable = gr.Checkbox(
                            config.enabled,
                            label="Enable",
                            elem_id=tab_prefix + "enable",
                        )
                        name = gr.Textbox(
                            config.name,
                            label="Worker Name",
                            elem_id=tab_prefix + "name",
                        )
                        allow_img2img = gr.Checkbox(
                            config.allow_img2img, label="Allow img2img"
                        )
                        allow_painting = gr.Checkbox(
                            config.allow_painting, label="Allow Painting"
                        )
                        allow_unsafe_ipaddr = gr.Checkbox(
                            config.allow_unsafe_ipaddr, label="Allow Unsafe IP Address"
                        )
                        allow_post_processing = gr.Checkbox(
                            config.allow_post_processing, label="Allow Post Processing"
                        )
                        restore_settings = gr.Checkbox(
                            config.restore_settings,
                            label="Restore settings after rendering a job",
                        )
                        nsfw = gr.Checkbox(config.nsfw, label="Allow NSFW")
                        interval = gr.Slider(
                            0,
                            60,
                            config.interval,
                            step=1,
                            label="Duration Between Generations (seconds)",
                        )
                        max_pixels = gr.Textbox(
                            str(config.max_pixels),
                            label="Max Pixels",
                            elem_id=tab_prefix + "max-pixels",
                        )
                        endpoint = gr.Textbox(
                            config.endpoint,
                            label="Stable Horde Endpoint",
                            elem_id=tab_prefix + "endpoint",
                        )
                        save_images_folder = gr.Textbox(
                            config.save_images_folder,
                            label="Folder to Save Generation Images",
                            elem_id=tab_prefix + "save-images-folder",
                        )

                    with gr.Box(scale=2):

                        def on_apply_selected_models(local_selected_models):
                            status.update(
                                f'Status: \
                            {"Running" if config.enabled else "Stopped"}, \
                            Updating selected models...'
                            )
                            selected_models = horde.set_current_models(
                                local_selected_models
                            )
                            local_selected_models_dropdown.update(
                                value=list(selected_models.values())
                            )
                            return f'Status: \
                            {"Running" if config.enabled else "Stopped"}, \
                            Selected models \
                            {list(selected_models.values())} updated'

                        local_selected_models_dropdown = gr.Dropdown(
                            [
                                model.name
                                for model in sd_models.checkpoints_list.values()
                            ],
                            value=[
                                model.name
                                for model in sd_models.checkpoints_list.values()
                                if model.name in list(config.current_models.values())
                            ],
                            label="Selected models for sharing",
                            elem_id=tab_prefix + "local-selected-models",
                            multiselect=True,
                            interactive=True,
                        )

                        local_selected_models_dropdown.change(
                            on_apply_selected_models,
                            inputs=[local_selected_models_dropdown],
                            outputs=[status],
                        )
                        gr.Markdown(
                            "Once you select a model it will take some time to load."
                        )

                with gr.Column():
                    show_images = gr.Checkbox(
                        config.show_image_preview, label="Show Images"
                    )
                    save_images = gr.Checkbox(config.save_images, label="Save Images")

                    refresh = gr.Button(
                        "Refresh", visible=False, elem_id=tab_prefix + "refresh"
                    )
                    refresh_image = gr.Button(
                        "Refresh Image",
                        visible=False,
                        elem_id=tab_prefix + "refresh-image",
                    )

                    state = gr.Textbox("", label="", readonly=True)

                    current_id = gr.Textbox(
                        "Current ID: ",
                        label="",
                        elem_id=tab_prefix + "current-id",
                        readonly=True,
                    )
                    preview = gr.Gallery(
                        label="Preview",
                        elem_id=tab_prefix + "preview",
                        visible=config.show_image_preview,
                        readonly=True,
                    ).style(grid=4)

                    def on_refresh(image=True, show_images=config.show_image_preview):
                        cid = horde.state.id
                        images = []

                        while True:
                            time.sleep(1.5)

                            if not config.enabled:
                                yield "Current ID: null", "", "Stopped", []

                            cid = horde.state.id
                            html = "".join(
                                map(
                                    lambda x: f"<p>{x[0]}: {x[1]}</p>",
                                    horde.state.to_dict().items(),
                                )
                            )
                            if image and show_images:
                                images = (
                                    [horde.state.image]
                                    if horde.state.image is not None
                                    else []
                                )
                            yield f"Current ID: {cid}", html, horde.state.status, images

                    with gr.Row():
                        log = gr.HTML(elem_id=tab_prefix + "log")

                    refresh_image.click(
                        fn=on_refresh,
                        outputs=[current_id, log, state, preview],
                        show_progress=False,
                    )
        apply_settings.click(
            fn=apply_stable_horde_settings,
            inputs=[
                enable,
                name,
                allow_img2img,
                allow_painting,
                allow_unsafe_ipaddr,
                allow_post_processing,
                restore_settings,
                nsfw,
                interval,
                max_pixels,
                endpoint,
                show_images,
                save_images,
                save_images_folder,
            ],
            outputs=[status, running_type],
        )

        return worker_ui


def get_user_ui():
    with gr.Blocks() as user_ui:
        gr.HTML(elem_id=f"{tab_prefix}user-html")
        return user_ui


def on_ui_tabs():
    with gr.Blocks() as demo:
        with gr.Row():
            apikey = gr.Textbox(
                config.apikey,
                label="Stable Horde API Key",
                elem_id=tab_prefix + "apikey",
            )
            save_apikey = gr.Button("Save", elem_id=f"{tab_prefix}apikey-save")

            def save_apikey_fn(apikey: str):
                config.apikey = apikey
                config.save()

            save_apikey.click(fn=save_apikey_fn, inputs=[apikey])

        with gr.Tab("Worker"):
            get_worker_ui()

        with gr.Tab("User"):
            get_user_ui()

    return ((demo, "Stable Horde Worker", "stable-horde"),)


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
