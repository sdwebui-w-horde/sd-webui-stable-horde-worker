let globalHordeTimer
let globalHordeCurrentId

function stableHordeStartTimer() {
  if (!globalHordeTimer) {
    globalHordeTimer = setInterval(() => {
      const currentId = gradioApp().querySelector('#stable-horde #stable-horde-current-id textarea')?.value
      const refreshBtn = gradioApp().querySelector('#stable-horde #stable-horde-refresh')
      if (refreshBtn) {
        refreshBtn.click()
      }
      if (currentId !== globalHordeCurrentId) {
        globalHordeCurrentId = currentId
        gradioApp().querySelector('#stable-horde #stable-horde-refresh-image').click()
      }
    }, 1000)
  }
}

function stableHordeStopTimer() {
  if (globalHordeTimer) {
    clearInterval(globalHordeTimer)
    globalHordeTimer = null
  }
}

stableHordeStartTimer()
