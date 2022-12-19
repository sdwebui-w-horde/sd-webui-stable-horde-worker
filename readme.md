# Stable Horde Worker Client for Stable Diffusion WebUI

An unofficial [Stable Horde](https://stablehorde.net/) worker client as a [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) extension.

## WIP

**This extension is still WORKING IN PROGRESS**, and is not ready for production use.

## Install

Run the following command in the root directory of your Stable Diffusion WebUI installation:

```bash
git clone https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker.git extensions/stable-horde-worker
```

Then, launch the Stable Diffusion WebUI. And then, go to `Settings` tab page, you would see the `Stable Horde` section.

![settings](./screenshots/settings.png)

Setup your `API key` and `Team name` from [Stable Horde](https://stablehorde.net/) and click the `Apply settings` buttons.

Click the `Enable` checkbox to enable the Stable Horde worker client.

### Existing issues

- `img2img` and `inpainting` is not supported yet
