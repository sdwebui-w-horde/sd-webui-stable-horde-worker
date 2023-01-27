import launch

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers", "diffusers")  # NSFW filter
    launch.run_pip("install aiohttp", "aiohttp")  # asynchroneous HTTP requests
