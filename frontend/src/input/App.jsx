import { useState, useEffect } from 'react'

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

export default function App() {
  /*
    Main app, runs everything below it. Lots of usefull variables are defined here
    appData is the main one, it contains most things the app will use to store data
    it was chosen to do this in one big object as importing 20 unique states and setters
    got a bit cumbersome. The drawback of this approac is its harder to set individual
    variables but in the end its still easier to do it this way. 
  */
  const [appData, setAppData] = useState(() => ({
    /* Behaviour variables */
    theme:                localStorage.getItem("dcm-studio-theme") === "light" ? "light" : "dark",
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

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", appData.theme)
    localStorage.setItem("dcm-studio-theme", appData.theme)
  }, [appData.theme])

  return (
    <div class="app_root">
      <Banner {...{appData, setAppData}}/>
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
