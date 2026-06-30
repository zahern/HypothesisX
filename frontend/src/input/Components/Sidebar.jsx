import "../All.css"
import { isDataModelComplete } from "./Sub-Components.jsx"

const STEPS = {
  estimation: ["Data & Model", "Run"],
  search:     ["Data & Model", "Search Params", "Run"],
}

export default function Sidebar({appData, setAppData}) {
  /*
    Sidebar component. Holds the mode switcher (Estimation/Search),
    the per-mode step list, and an info box describing the current step.

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  const steps = STEPS[appData.mode]
  const locked = !isDataModelComplete(appData)

  // set mode helper to straiten out table issues
  const setMode = (mode) => {
    setAppData({...appData,
      ["mode"]: mode, step: 0,
      variables: appData.variables.map(v => {
        if (!v.include) {
          return {...v, always:false, boxcox:false, seCorr:false, random:false, dist:"none", estCorr:false, always:false}
        } else {
          return v
        }
      })
    })
  }

  return (
    <div class="sidebar">
      {/* Mode switching (Estimation/Search) */}
      <div class="sidebar_mode_group">
        <div key="estimation"
          onClick={() => setMode("estimation")}
          class={`mode_button ${appData.mode==="estimation" ? "mode_button_select" : ""}`}>
          Estimation
        </div>
        <div key="search"
          onClick={() => setMode("search")}
          class={`mode_button ${appData.mode==="search" ? "mode_button_select" : ""}`}>
          Search
        </div>
      </div>

      <div class="sidebar_head">
        {/* Info box */}
        <div class="sidebar_info_box">
          File: {appData.file.name ? appData.file.name : "No file Uploaded"} <br />
        </div>
      </div>

      {/* Steps */}
      <div class="sidebar_steps">
        {steps.map((s, i) => {
          const disabled = locked && i >= 1
          return (
            <div key={s}
              onClick={disabled ? undefined : () => setAppData({...appData, step:i})}
              class={`sidebar_div ${i===appData.step ? "sidebar_selected" : ""} ${disabled ? "sidebar_div_disabled" : ""}`}>
              <div class={`sidebar_circle ${i===appData.step ? "sidebar_circle_select" : ""} ${i<appData.step ? "sidebar_circle_done" : ""}`}>
                {i<appData.step ? "✓" : i+1}
              </div>
              <span class={`sidebar_label ${i===appData.step ? "sidebar_label_active" : ""} ${i<appData.step ? "sidebar_label_done" : ""}`}>{s}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
