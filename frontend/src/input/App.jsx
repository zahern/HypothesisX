import { useState } from 'react'

import Banner from "./Components/Banner.jsx"
import Sidebar from "./Components/Sidebar.jsx"
import Estimation from "./Components/Estimation.jsx"
import Search from "./Components/Search.jsx"

const DIST_OPTIONS = [
  { value: "none", label: "Fixed" },
  { value: "diff", label: "All..."},
  { value: "n",    label: "Normal (n)" },
  { value: "ln",   label: "Log-Normal (ln)" },
  { value: "u",    label: "Uniform (u)" },
  { value: "t",    label: "Triangular (t)" },
  { value: "tn",   label: "Truncated Normal (tn)" },
];

export default function App({ theme, setTheme }) {
  /*
    Input sub-app (the former DCM Studio). The `mode` / `setMode` and
    `theme` / `setTheme` props are owned by the unified top-level App so the
    Input/Output and Dark/Light toggles can be driven from this banner.
  */
  const [appData, setAppData] = useState(() => ({
    /* Behaviour variables */
    mode:                 "estimation",
    step:                 0,
    uploaded:             false,
    file:                 {name:"", rows:0, cols:0},

    /* Model and data choices variables */
    model:                "MXL",
    mixedModels:          ["MXL", "MXRRM"],
    columnUniques:        {},
    choiceId:             "Upload Data to Detect",
    indId:                "Upload Data to Detect",
    altVar:               "Upload Data to Detect",
    choiceVar:            "Upload Data to Detect",
    choiceSet:            "Upload Data to Detect",
    avVar:                "Upload Data to Detect",
    distrabutionOptions:  DIST_OPTIONS,
    variables:            [],
    baseAlt:              "None",
    fitIntercept:         false,
    gtol:                 "1e-5",

    /* Estimation exclusive variables*/
    ftol:                 "1e-8",
    running:              false,
    done:                 false,
    draws:                1000,

    /* Search exclusive variables*/
    criterion:            "bic",
    allowRandom:          true,
    allowCorr:            true,
    allowBC:              true,
    algorithm:            "Simulated Annealing",
    runName:              "SIMAN_BERLIN",
    steps:                15,
    iterations:           60,

    /* Live search progress (driven by /ws/progress) */
    progress:         { current: 0, total: 0, status: "idle" }, // status: "idle" | "running" | "done"
  }))

  const [csv, setCsv]             = useState()

  return (
    <div class="app_root">
      <Banner {...{appData, setAppData, theme, setTheme}}/>
      <div class="main_layout">
        <Sidebar {...{appData, setAppData}} />
        <div class="content">
          {appData.mode === "estimation" && <Estimation {...{appData, setAppData, csv, setCsv}} />}
          {appData.mode === "search" && <Search {...{appData, setAppData, csv, setCsv}} />}
        </div>
      </div>
    </div>
  )
}
