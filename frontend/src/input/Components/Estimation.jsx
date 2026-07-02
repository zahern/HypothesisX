import { useState, useEffect } from 'react'

import { Grid, SelectField, Spinner, PageButtons, CSVUploadBox, DataOptions, ModelOptions, ModelSelect, SummaryBox, DoneMsg, isDataModelComplete} from "./Sub-Components.jsx"
import { VariableOptions, RunTolerance, RunButton, ExportButton } from "./Estimation-Sub-Components.jsx"

import "../All.css"

export default function Estimation({appData, setAppData, csv, setCsv}) {
  /*
    Main function of this file, returns basic stuff and the further steps in the process

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    csv         -> state to store the csv file
    setCsv      -> function to set the state of csv
  */

  const [estRun, setEstRun] = useState({status: "idle", bic:0, ll:0, aic:0})

  return (
    <>
      {appData.step === 0 && <Step0 {...{appData, setAppData, csv, setCsv}} />}
      {appData.step === 1 && <Step1 {...{appData, setAppData, csv, estRun, setEstRun}}/>}
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
    
      {/* ------------------------- Variable Options ------------------------- */}
      <h2>Variable Configuration</h2>
      <p>BoxCox always available. Correlated only for Mixed + Random. BoxCox ↔ Correlated are mutually exclusive.</p>
    
      <VariableOptions {...{appData, setAppData}} />

      <PageButtons {...{appData, setAppData}} canNext={isDataModelComplete(appData)} next={true} back={false} />

    </div>
  )
}

function Step1({appData, setAppData, csv, estRun, setEstRun}) {
  /*
    2nd step, handles finalizing and running the model
    
    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    csv         -> state to store the csv file
  */
  // helpers
  const listVars = (f) => {return appData.variables.filter(f).map(v => v.name).join(", ")}
  const nVar = (f) => {return appData.variables.filter(f).length}
  
  return (
    <div>
      { estRun.status === "idle" && (
        <div>
          <h2>Run Estimation</h2>
          <RunTolerance {...{appData, setAppData}} />

          <SummaryBox rows={[
            ["Model", appData.model], 
            ["Run Name", appData.runName],
            ["Data", `${appData.file.name} · ${appData.file.rows} rows · ${appData.file.cols} columns`],
            ["Base Alternative", appData.baseAlt],
            ["Variables", `${appData.fitIntercept ? "Fit Intercept" : " No Intercept"} · ${appData.variables.length} total || ${nVar(v => v.dist==="none" && v.include)} fixed · ${nVar(v => v.dist!=="none" && v.include)} random · ${nVar(v => v.estCorr)} correlated · ${nVar(v => v.boxcox)} boxcox · ${nVar(v => !v.include)} excluded`],
            ["Fixed", listVars(v => v.dist==="none" && v.include) || "None"],
            ["Random", listVars(v => v.dist!=="none" && v.include) || "None"],
            ["Correlated", listVars(v => v.estCorr && v.include) || "None"],
            ["BoxCox", listVars(v => v.boxcox && v.include) || "None"],                
            ["Excluded", listVars(v => !v.include) || "None"],                
            ["gtol / ftol", `${appData.gtol} / ${appData.ftol}`],
            ["Draws", appData.draws]
          ]} />
          <div class="button_row">
            <RunButton {...{appData, setAppData, csv, estRun, setEstRun}} />
            <ExportButton {...{appData}} />
          </div>
          <PageButtons {...{appData, setAppData}} canNext={false} next={false} back={true} />
        </div>
      )} {estRun.status==="running" && (
        <div>
          <Spinner label="Running Estimation..." />
        </div>
      )} {estRun.status==="done" && (
        <div>
          <DoneMsg label={`Estimation completed · LL: ${estRun.ll} · BIC: ${estRun.bic} · AIC: ${estRun.aic}`} />
        </div>
      )}
    </div>
  )
}
