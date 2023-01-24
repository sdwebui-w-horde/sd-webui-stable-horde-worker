<p align="center">
  <img src="./logo.png" width="256px"></img>
</p>

<div align="center">

# SD WebUI ❤️ Stable Horde

✨ *Stable Horde Worker Bridge for Stable Diffusion WebUI* ✨

</div>

An unofficial [Stable Horde](https://stablehorde.net/) worker bridge as a [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) extension.

## Features

**This extension is still WORKING IN PROGRESS**, and is not ready for production use.

- Get jobs from Stable Horde, generate images and submit generations
- Configurable interval between every jobs
- Enable and disable extension whenever

## Install

Run the following command in the root directory of your Stable Diffusion WebUI installation:

```bash
git clone https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker.git extensions/stable-horde-worker
```

Then, launch the Stable Diffusion WebUI. And then, go to `Settings` tab page, you would see the `Stable Horde` section.

![settings](./screenshots/settings.png)

Setup your `API key` and `Worker name` from [Stable Horde](https://stablehorde.net/) and click the `Apply settings` buttons.

Click the `Enable` checkbox to enable the Stable Horde worker client.

### Existing issues

- only one model is usable
- [model should be selected manually](https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker/issues/3)
