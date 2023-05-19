let globalHordeTimer
let globalHordeCurrentId

function stableHordeStartTimer() {
  if (!globalHordeTimer) {
    globalHordeTimer = setInterval(() => {
      const currentId = gradioApp().querySelector('#stable-horde-current-id textarea')?.value
      const refreshBtn = gradioApp().querySelector('#stable-horde-refresh')
      if (refreshBtn) {
        refreshBtn.click()
      }
      if (currentId !== globalHordeCurrentId) {
        globalHordeCurrentId = currentId
        gradioApp().querySelector('#stable-horde-refresh-image').click()
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

async function stableHordeSwitchMaintenance(id) {
  const r = await fetch(`/stable-horde/workers/${id}`, {
    method: 'PUT',
    body: JSON.stringify({
      maintenance: true,
    }),
    headers: {
      'Content-Type': 'application/json',
    },
  })
  const data = await r.json()
  if (typeof data.maintenance !== 'undefined') {
    console.log('maintenance', data.maintenance)
  }
}

;(() => {
  const timer = setInterval(() => {
    const refresh = gradioApp().querySelector('#stable-horde-refresh-image')
    if (refresh) {
      clearInterval(timer)
      refresh.click()
    }
  }, 1000)
})()
