import { useState, useEffect } from 'react'

import { Grid, SelectField, Spinner, PageButtons, CSVUploadBox, DataOptions, ModelOptions, ModelSelect, SummaryBox, isDataModelComplete} from "./Sub-Components.jsx"
import { SearchAllow, VariableOptions, SearchParams, Metaheristic, RunButton, ExportButton } from "./Search-Sub-Components.jsx"

import "../All.css"

export default function Estimation({appData, setAppData, csv, setCsv}) {
  /*
    Main function of this file, returns basic stuff and the further steps in the process

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    csv         -> state to store the csv file
    setCsv      -> function to set the state of csv
  */

  const [saRun, setSaRun] = useState({status: "idle", complete:0, total:500})

  return (
    <>
      {appData.step === 0 && <Step0 {...{appData, setAppData, csv, setCsv}} />}
      {appData.step === 1 && <Step1 {...{appData, setAppData}}/>}
      {appData.step === 2 && <Step2 {...{appData, setAppData, csv, saRun, setSaRun}}/>}
    </>
  )
}

function Step0({appData, setAppData, csv, setCsv}) {
  /*
    1st step, handles data, model, and variable config
    
    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    csv         -> state to store the csv file
    setCsv      -> function to set the state of csv
  */
  // helper to quickly generate functions to write to appData
  const sapp = (key) => {
    return (val) => setAppData({...appData, [key]: val})
  };

  return (
    <div>
      <h2>Data Input</h2>

      {/* ------------------------- Data upload ------------------------- */}
      <p>Upload your panel data CSV. Columns detected automatically.</p>
      <CSVUploadBox {...{appData, setAppData, csv, setCsv, }} />
      
      {/* ------------------------- Data Options ------------------------- */}
      <DataOptions {...{appData, setAppData}} />
      
      {/* ------------------------- Model Options ------------------------- */}
      <h2>Model Structure</h2>
  
      <ModelOptions {...{appData, setAppData}} />
      <ModelSelect {...{appData, setAppData}} />
      <SearchAllow {...{appData, setAppData}} />
    
      {/* ------------------------- Variable Options ------------------------- */}
      <h2>Variable Configuration</h2>
      <p>BoxCox always available. Correlated only for Mixed + Random. BoxCox ↔ Correlated are mutually exclusive.</p>
    
      <VariableOptions {...{appData, setAppData}} />

      <PageButtons {...{appData, setAppData}} canNext={isDataModelComplete(appData)} next={true} back={false} />

    </div>
  )
}

function Step1({appData, setAppData}) {
  /*
    2nd step, handles configuring search parameters
    
    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  return (
    <div>
      <h2>Meteheristic</h2>
      <Metaheristic {...{appData, setAppData}} />

      <h2>Search Parameters</h2>
      <p>Configure the {appData.algorithm} search.</p>
      <SearchParams {...{appData, setAppData}} />
      
      <PageButtons {...{appData, setAppData}} canNext={true} next={true} back={true} />
    </div>
  )
}

function Step2({appData, setAppData, csv, saRun, setSaRun}) {
  /*
    3rd step, handles finalizing and running the model
    
    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    csv         -> state to store the csv file
  */
  // helpers
  const listVars = (f) => {return appData.variables.filter(f).map(v => v.name).join(", ")}
  const nVar = (f) => {return appData.variables.filter(f).length}

  return (
    <div>
      { saRun.status === "idle" && (<> 
        <h2>Run Search</h2>
        <SummaryBox rows={[
          ["Model", appData.model],
          ["Run Name", appData.runName],
          ["Data", `${appData.file.name} · ${appData.file.rows} rows · ${appData.file.cols} columns`],
          ["Base Alternative", appData.baseAlt],
          ["Variables", `${appData.fitIntercept ? "Fit Intercept" : " No Intercept"} · ${appData.variables.length} total || ${nVar(v => !v.random && v.include)} fixed · ${nVar(v => v.random && v.include)} random · ${nVar(v => v.seCorr)} correlated · ${nVar(v => v.boxcox)} boxcox · ${nVar(v => !v.include)} excluded · ${nVar(v => v.always)} always included`],
          ["Fixed", listVars(v => !v.random) || "None"],
          ["Random", listVars(v => v.random) || "None"],
          ["Correlated", listVars(v => v.seCorr) || "None"],
          ["BoxCox", listVars(v => v.boxcox) || "None"],
          ["Excluded", listVars(v => !v.include) || "None"],
          ["Always Included", listVars(v => v.always) || "None"],
          ["gtol", `${appData.gtol}`],
          ["Algorithm", appData.algorithm],
          ["Steps", `${appData.steps}`],
          ["Iterations", `${appData.iterations}`],
          ["Draws", appData.draws],
          ["Criterion", appData.criterion.toUpperCase()],
        ]} />
        <div class="button_row">
          <RunButton {...{appData, setAppData, csv, saRun, setSaRun}} />
          <ExportButton {...{appData}} />
        </div>
        <PageButtons {...{appData, setAppData}} canNext={false} next={false} back={true} />

      </>)} {saRun.status === "running" && (<>
        <Spinner label="Running Search..." />
        <div class="var_summary_box">{`Tried ${saRun.complete} / ${saRun.total} possibilities`}</div>

      </>)} {saRun.status === "done" && (<>
        <div class="var_summary_box">{`Finished Search!`}</div>
      </>)}
    </div>
  )
}
