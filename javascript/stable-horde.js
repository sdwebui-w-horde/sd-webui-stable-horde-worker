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

// stableHordeStartTimer()

async function stableHordeSwitchMaintenance(e) {
  const url = e.target.dataset.api;
  e.target.disabled = true;
  e.target.innerText = "Switching...";

  const response = await fetch(url, {
      method: "PUT",
      headers: {
          "apikey": "{apikey}",
          "Content-Type": "application/json"
      },
      body: JSON.stringify({
          "maintenance_mode": true
      })
  });

  if (response.ok) {
      e.target.innerText = "Switched";
      setTimeout(() => {
          e.target.disabled = false;
          e.target.innerText = "Switch Maintenance";
      }, 1000);
  } else {
      e.target.innerText = "Failed";
      setTimeout(() => {
          e.target.disabled = false;
          e.target.innerText = "Switch Maintenance";
      }, 1000);
  }
}

;(() => {
  const timer = setInterval(() => {
    const refresh = gradioApp().querySelector('#stable-horde #stable-horde-refresh-image')
    if (refresh) {
      clearInterval(timer)
      refresh.click()
    }
  }, 1000)
})()
