<p align="center">
  <img src="./logo.png" width="256px"></img>
</p>

<div align="center">

# SD WebUI ❤️ Stable Horde

![python](https://img.shields.io/badge/python-3.10-blue)
[![issues](https://img.shields.io/github/issues/sdwebui-w-horde/sd-webui-stable-horde-worker)](https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker/issues)
[![pr](https://img.shields.io/github/issues-pr/sdwebui-w-horde/sd-webui-stable-horde-worker)](https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker/pulls)
[![license](https://img.shields.io/github/license/sdwebui-w-horde/sd-webui-stable-horde-worker)](LICENSE)
[![Lint](https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker/actions/workflows/lint.yml/badge.svg)](https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker/actions/workflows/lint.yml)

✨ *Stable Horde Worker Bridge for Stable Diffusion WebUI* ✨

</div>

An unofficial [Stable Horde](https://stablehorde.net/) worker bridge as a [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) extension.

## Features

**This extension is still WORKING IN PROGRESS**, and is not ready for production use.

- Get jobs from Stable Horde, generate images and submit generations
- Configurable interval between every jobs
- Enable and disable extension whenever
- Detect current model and fetch corresponding jobs on the fly
- Show generation images in the Stable Diffusion WebUI
- Save generation images with png info text to local

## Install

- Run the following command in the root directory of your Stable Diffusion WebUI installation:

  ```bash
  git clone https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker.git extensions/stable-horde-worker
  ```

- Launch the Stable Diffusion WebUI, You would see the `Stable Horde Worker` tab page.

  ![settings](./screenshots/settings.png)

- Register an account on [Stable Horde](https://stablehorde.net/) and get your `API key` if you don't have one.

  **Note**: the default anonymous key `00000000` is not working for a worker, you need to register an account and get your own key.

- Setup your `API key` here.
- Setup `Worker name` here with a proper name.
- Make sure `Enable` is checked.
- Click the `Apply settings` buttons.


## License

This project is licensed under the terms of the [AGPL-3.0 License](LICENSE).
