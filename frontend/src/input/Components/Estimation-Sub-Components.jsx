import { useState, useEffect } from 'react'

import { Grid, SelectField, SliderField } from "./Sub-Components.jsx"
import { VarTable, VarTableHeader, VarTableRow } from "./VarTable.jsx"


export function VariableOptions({appData, setAppData}) {
  /*
    Function for generating the variable options table. This function has a lot in it
    the code is quite ugly and held together by prayers, but it was chosen to impliment
    all the functionality of the table in one function. Previously it was controlled
    with several control variables that were set in miscellanious functions spread
    throughout the codebase. Implementing everything here is ugly but means its all together

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  // helper to quickly generate functions to write to appData
  const sapp = (key) => {
    return (val) => setAppData({...appData, [key]: val})
  };

  const isMixed = appData.mixedModels.includes(appData.model)
  const includeDisabled = appData.variables.length === 0
  const noneIncluded = appData.variables.every((v) => !v.include)
  const boxcoxDisabled = appData.variables.length === 0 || noneIncluded
  const corrDisabled = appData.variables.every((v) => v.dist==="none") || !isMixed || appData.variables.length === 0 || noneIncluded

  const cols = [
    { label: "Valriable",    className: "col_name" },
    { label: "Distrabution", className: "col_dist" },
    { label: "Include",      className: "col_check" },
    { label: "BoxCox",       className: "col_check" },
    { label: "Correlated",   className: "col_check" },
  ]

  const allSame = appData.variables.length > 0 && appData.variables.every(v => v.dist === appData.variables[0].dist)
  const headerDist = appData.variables.length === 0 ? appData.distrabutionOptions[0].value : allSame ? appData.variables[0].dist : "diff"

  const bulkCells = [
    null,
    <SelectField
      value={headerDist}
      onChange={(d) => {
        if (d === "diff") return;
        setAppData({...appData,
          variables: appData.variables.map(v => ({
            ...v,
            dist: d,
            estCorr: d === "none" ? false : v.estCorr,
          })),
        });
      }}
      options={appData.distrabutionOptions.filter(o => o.value !== "diff" || headerDist === "diff")}
      disabled={!isMixed || includeDisabled}
    />,
    <input type="checkbox" checked={
      !appData.variables.some((v) => !v.include) && ! includeDisabled
    } onClick={(e) => {setAppData({...appData,
      variables: appData.variables.map((v) => {return {...v, include:e.target.checked, boxcox:false, estCorr:false}})
    })}} disabled={includeDisabled}/>,
    <input type="checkbox" checked={
      appData.variables.every(v => !v.include || v.boxcox) && !boxcoxDisabled
    } onClick={(e) => {setAppData({...appData,
      variables: appData.variables.map((v) => {
        return !v.include ? v : {...v, boxcox:e.target.checked, estCorr:false}
      })
    })}} disabled={boxcoxDisabled}/>,
    <input type="checkbox" checked={!appData.variables.some((v, i, arr) => {
      return v.dist !== "none" && !v.estCorr
    }) && !corrDisabled
    } onClick={(e) => {setAppData({...appData,
      variables: appData.variables.map((v, i) => {
        return v.dist==="none" ? v : {...v, boxcox:false, estCorr:e.target.checked}
      })
    })}} disabled={corrDisabled}/>,
  ]

  return (
    <VarTable legend={
        <>
          <div class="table_dot" /><span class="table_text">Not Included</span>
          <div class="table_dot td_blue" /><span class="table_text">Included</span>
          <div class="table_dot td_green" /><span class="table_text">Random Variable</span>
          <span class="table_text">📦</span><span class="table_text">BoxCox</span>
          <span class="table_text">🔗</span><span class="table_text">Correlated</span>
        </>
      } summary={
        <>
          {appData.variables.filter(v=>v.dist==="none" && v.include).length} fixed · {appData.variables.filter(v=>v.dist!=="none" && v.include).length} random · {appData.variables.filter(v=>v.boxcox).length} boxcox · {appData.variables.filter(v=>v.estCorr).length} correlated · {appData.variables.filter(v=>!v.include).length} excluded
        </>
      }>
      <VarTableHeader cols={cols} bulkCells={bulkCells} />
      <tbody class="tbody">

        {/* Loop through every row in the variable storeage and do ugly things to create table rows for them all */}

        {appData.variables.map((variable, i) => (
          <VarTableRow
            key={i}
            variable={variable}
            dotBlue={variable.include}
            dotGreen={variable.dist!=="none" && variable.include}
            prefix={`${variable.boxcox ? "📦" : ""}${variable.estCorr ? "🔗" : ""}`}
            cells={[

              <td> {/* Select distrabution */}
                <select value={variable.dist} disabled={!isMixed || !variable.include} onChange={(e) => {
                    const d = e.target.value // this select is very ugly, maybe split it off into it's own thing
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {...v,
                        dist: d,
                        estCorr: d==="none" ? false : v.estCorr,
                      } : v})]
                    })
                  }} class={`generic_input var_dist_select ${!isMixed || !variable.include ? "disabled_input" : ""} ${variable.dist === "none" ? "" : "var_dist_select_active"}`}>
                  {appData.distrabutionOptions.map(o => (
                    o.value !== "diff" && <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </td>,

              <td class="center"> {/* Included checkbox */}
                <input type="checkbox" checked={variable.include} onChange={(e) => {
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {
                        ...v, 
                        include:!v.include, 
                        boxcox:false, 
                        estCorr:false
                      } : v})]
                    })
                }} />
              </td>,
              <td class="center"> {/* BoxCox checkbox */}
                <input type="checkbox" checked={variable.boxcox} disabled={!variable.include || boxcoxDisabled} onChange={(e) => {
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {
                        ...v, 
                        boxcox:!v.boxcox, 
                        estCorr:false
                      } : v})]
                    })
                }} />
              </td>,
              <td class="center"> {/* Correlation checkbox */}
                <input type="checkbox" checked={variable.estCorr} disabled={variable.dist==="none" || corrDisabled || !variable.include} onChange={(e) => {
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {
                        ...v, 
                        boxcox:false, 
                        estCorr:!v.estCorr
                      } : v})]
                    })
                }} />
              </td>,
            ]}
          />
        ))}
      </tbody>
    </VarTable>
  )
}

export function RunTolerance({appData, setAppData}) {
  /*
    component to change the tolerences of the run and the draws
    
    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  // helper to quickly generate functions to write to appData
  const sapp = (key) => {
    return (val) => setAppData({...appData, [key]: val})
  };

  return (
    <div class="section_mb_24">

      <div class="field_row_mb_20">
        <div class="label">Run Name</div>
        <input class="generic_input" value={appData.runName} onChange={e => sapp("runName")(e.target.value)} />
      </div>

      <Grid cols={2}>
        <div>
          <SelectField 
            label="gtol" 
            value={appData.gtol} 
            onChange={sapp("gtol")} 
            options={["1e-3","1e-4","1e-5","1e-6","1e-7","1e-8","1e-9"]} 
            hint="Gradient tolerance" 
          />
        </div>
        <div>
          <SelectField 
            label="ftol" 
            value={appData.ftol} 
            onChange={sapp("ftol")} 
            options={["1e-6","1e-7","1e-8","1e-9","1e-10"]} 
            hint="Funciton value tolerance" 
          />
        </div>
        <div>
        <SliderField label="Draws (R)" value={appData.draws} min={100} max={3000} step={100} onChange={sapp("draws")} hint="Simulation draws" />
        </div>
      </Grid>
    </div>
  )
}

function generatePy(appData) {
  /*
    Tempory legacy functionality for exporting python files directly

    appData -> the full app state object from App.jsx
  */
  const included      = appData.variables.filter(v => v.include);
  const isRRM         = appData.model === "RRM" || appData.model === "MXRRM";
  const modelClass    = isRRM ? "MixedRandomRegret" : "MixedLogit";
  const modelImport   = isRRM
    ? `from SearchLibrium.MixedRandomRegret import MixedRandomRegret`
    : `from SearchLibrium.mixed_logit import MixedLogit`;
  const panelDisabled = !appData.mixedModels.includes(appData.model);
  const fileName      = appData.file.name;

  const varList   = included.map(v => `'${v.name}'`).join(", ");
  const randvars  = included.filter(v => v.dist !== "none");
  const transvars = included.filter(v => v.boxcox);
  const corvars   = included.filter(v => v.estCorr);
  const randDict  = randvars.map(v => `'${v.name}': '${v.dist}'`).join(", ");
  const transStr  = transvars.map(v => `'${v.name}'`).join(", ");
  const corrStr   = corvars.map(v => `'${v.name}'`).join(", ");

  return `import pandas as pd
import sys, numpy as np, random, os
sys.path.insert(0, "/home/tacomora/SearchLibrium")
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["TF_DETERMINISM_OPS"] = "1"

${modelImport}

df = pd.read_csv('${fileName || "data.csv"}')

varnames   = [${varList}]
choice_id  = df['${appData.choiceId}']
ind_id     = df['${appData.indId}']
choice_var = df['${appData.choiceVar}']
alt_var    = df['${appData.altVar}']
choice_set = [${appData.choiceSet.split(",").map(s => `'${s.trim()}'`).join(", ")}]
base_alt   = ${appData.baseAlt === "None" ? "None" : `'${appData.baseAlt}'`}
R          = ${appData.draws}
gTol       = ${appData.gtol}
fTol       = ${appData.ftol}

model = ${modelClass}()
model.setup(
    X=df[varnames],
    y=choice_var,
    varnames=varnames,
    ids=choice_id,
    panels=${panelDisabled ? "None" : "ind_id"},
    alts=alt_var,
    base_alt=base_alt,
    fit_intercept=${appData.fitIntercept ? "True" : "False"},
    n_draws=R,
    avail=${appData.avVar !== "None" ? `df['${appData.avVar}']` : "None"},
    gtol=gTol,
    ftol=fTol,${randvars.length ? `\n    randvars={${randDict}},` : ""}${transvars.length ? `\n    transvars=[${transStr}],` : ""}${corvars.length ? `\n    correlated_vars=[${corrStr}],` : ""}
)
model.fit()
model.summarise()
`;
}

export function ExportButton({appData}) {
  /*
    Tempory legacy functionality for exporting python files directly
  
    appData -> state holding most data for the app
  */
  const handleExport = () => {
    const blob = new Blob([generatePy(appData)], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `EST_${appData.model}_${appData.file.name || "model"}.py`;
    a.click();
  };

  return <button onClick={handleExport} class="export_button">↓ Export .py</button>
}

async function handleRun(appData, setAppData, csv, estRun, setEstRun) {
  /*
    Send the current appData and the uploaded csv to the backend, Does 
    nothing for now, but starts up comunication. Doesn't handle many of the isues
    that will occur eventually, when a proper backend is implimented.

    It might be advisable in the future to use useEffect and/or other react
    components to mek this more relable, I've seen the package
    react-use-websocket be recommended online.

    appData    -> the full app state object from App.jsx
    csv        -> the File object stored in App.jsx via setCsv (from CSVUploadBox)
    setAppData -> state setter, used to seed progress.total from the response
  */
  if (!csv) throw new Error("No CSV file loaded — upload a CSV before running.")

  const proto = window.location.protocol === "https:" ? "wss:" : "ws:"
  const socket = new WebSocket(`${proto}//${window.location.host}/ws/run_estimation`)

  // Connection opened
  socket.addEventListener("open", event => {
    setEstRun({...estRun, status: "running"})
    socket.send(JSON.stringify(appData))
    socket.send(csv) // sends as blob
  });

  // Listen for messages
  socket.addEventListener("message", event => {
    const msg = JSON.parse(event.data)
    setEstRun({...estRun, 
      status: msg.status ?? "done",
      ll:  msg.ll ?? 0,
      bic: msg.bic ?? 0,
      aic: msg.aic ?? 0,
    })
  });
}

export function RunButton({appData, setAppData, csv, estRun, setEstRun}) {
  /*
    Button that runs the model
    
    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    csv         -> state to store the csv file
  */

  return (
    <button onClick={() => handleRun(appData, setAppData, csv, estRun, setEstRun)} class="run_button">▶ Run Estimation</button>
  )
}
