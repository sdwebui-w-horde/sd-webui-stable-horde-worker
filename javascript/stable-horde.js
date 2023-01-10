var globalHordeTimer

function toggle(enable) {
  const enableBtn = gradioApp().querySelector('#stable-horde #stable-horde-enable')
  const disableBtn = gradioApp().querySelector('#stable-horde #stable-horde-disable')
  if (enable && enableBtn) {
    enableBtn.style.display = 'none'
    disableBtn.style.display = 'flex'
  }
  if (!enable && disableBtn) {
    disableBtn.style.display = 'none'
    enableBtn.style.display = 'flex'
  }
}

function toggleEnable() {
  toggle(true)
  if (!globalHordeTimer) {
    globalHordeTimer = setInterval(() => {
      const hordeBtn = gradioApp().querySelector('#stable-horde #stable-horde-refresh')
      if (hordeBtn) {
        hordeBtn.click()
      }
    }, 1000)
  }
}

function toggleDisable() {
  toggle(false)
  if (globalHordeTimer) {
    clearInterval(globalHordeTimer)
    globalHordeTimer = null
  }
}
