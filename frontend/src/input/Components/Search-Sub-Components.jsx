import { Grid, SelectField, SliderField, Toggle } from "./Sub-Components.jsx"
import { VarTable, VarTableHeader, VarTableRow } from "./VarTable.jsx"

import "../All.css"

export function SearchAllow({appData, setAppData}) {
  /*
    component to allow or disallow various togles for the search
    
    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  const isMixed = appData.mixedModels.includes(appData.model)

  return (
    <div class="section_mt_24">
      <div class="label">Allow in Search</div>
      <Grid cols={2} style={{ marginTop:10 }}>
        <Toggle label="Random Parameters" active={appData.allowRandom}  onChange={v => { setAppData({
          ...appData,
          allowRandom: v,
          allowCorr: false,
          variables: appData.variables.map((a) => {return {...a, random:!v ? false : a.random, seCorr:false}}),
        })}} />
        <Toggle label="Correlated Varibales" active={appData.allowCorr}  onChange={v => { setAppData({
          ...appData,
          allowCorr: appData.allowRandom ? v : false,
          variables: appData.variables.map((a) => {return {...a, seCorr:!v ? false : a.seCorr}}),
        })}} />
        <Toggle label="Box-Cox Variables" active={appData.allowBC}  onChange={v => { setAppData({
          ...appData,
          allowBC: v,
          variables: appData.variables.map((a) => {return {...a, boxcox:!v ? false : a.boxcox}}),
        })}} />
      </Grid>
    </div>
  )
}

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
  const alwaysDisabled = appData.variables.length === 0 || noneIncluded
  const boxcoxDisabled = appData.variables.length === 0 || noneIncluded || !appData.allowBC
  const randomDisabled = appData.variables.length === 0 || noneIncluded || !appData.allowRandom
  const corrDisabled = appData.variables.every((v) => !v.random) || !isMixed || appData.variables.length === 0 || noneIncluded || ! appData.allowCorr

  const cols = [
    { label: "Valriable",      className: "col_name_wide" },
    { label: "Include",        className: "col_check" },
    { label: "Always Include", className: "col_check" },
    { label: "BoxCox",         className: "col_check" },
    { label: "Random",         className: "col_check" },
    { label: "Correlated",     className: "col_check" },
  ]

  const bulkCells = [
    null,
    /* Include */
    <input type="checkbox" checked={
      !appData.variables.some((v) => !v.include) && ! includeDisabled
    } onClick={(e) => {setAppData({...appData,
      variables: appData.variables.map((v) => {return {...v, include:e.target.checked, always:false, boxcox:false, random:false, seCorr:false}})
    })}} disabled={includeDisabled}/>,
    /* Always Include */
    <input type="checkbox" checked={
      appData.variables.every(v => !v.include || v.always) && !alwaysDisabled
    } onClick={(e) => {setAppData({...appData,
      variables: appData.variables.map((v) => {
        return !v.include ? v : {...v, always:e.target.checked}
      })
    })}} disabled={alwaysDisabled}/>,
    /* BoxCox */
    <input type="checkbox" checked={
      appData.variables.every(v => !v.include || v.boxcox) && !boxcoxDisabled
    } onClick={(e) => {setAppData({...appData,
      variables: appData.variables.map((v) => {
        return !v.include ? v : {...v, boxcox:e.target.checked, seCorr:false}
      })
    })}} disabled={boxcoxDisabled}/>,
    /* Random */
    <input type="checkbox" checked={
      appData.variables.every(v => !v.include || v.random) && isMixed && !randomDisabled
    } onClick={(e)=> {setAppData({...appData,
      variables: appData.variables.map((v) => {
        return !v.include ? v : {...v, random:e.target.checked, seCorr:false}
      })
    })}} disabled={randomDisabled || !isMixed}/>,
    /* Correlation */
    <input type="checkbox" checked={!appData.variables.some((v, i, arr) => {
      return v.random && !v.seCorr
    }) && !corrDisabled
    } onClick={(e) => {setAppData({...appData,
      variables: appData.variables.map((v, i) => {
        return !v.random ? v : {...v, boxcox:false, seCorr:e.target.checked}
      })
    })}} disabled={corrDisabled}/>,
  ]

  return (
    <VarTable legend={
    <>
      <div class="table_dot" /><span class="table_text">Not Included</span>
      <div class="table_dot td_blue" /><span class="table_text">Included</span>
      <span class="table_text td_blue_text">Always Included</span>
      <div class="table_dot td_green" /><span class="table_text">Random Variable</span>
      <span class="table_text">📦</span><span class="table_text">BoxCox</span>
      <span class="table_text">🔗</span><span class="table_text">Correlated</span>
    </>
      } summary={
    <>
      {appData.variables.filter(v=>!v.random && v.include).length} fixed · {appData.variables.filter(v=>v.random && v.include).length} random · {appData.variables.filter(v=>v.boxcox).length} boxcox · {appData.variables.filter(v=>v.seCorr).length} correlated · {appData.variables.filter(v=>!v.include).length} excluded · {appData.variables.filter(v=>v.always).length} always included
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
            dotGreen={variable.random}
            textClass={variable.always ? "td_blue_text" : ""}
            prefix={`${variable.boxcox ? "📦" : ""}${variable.seCorr ? "🔗" : ""}`}
            cells={[
              /* Include */
              <td class="center">
                <input type="checkbox" checked={variable.include} onChange={(e) => {
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {...v, include:!v.include, always:false, boxcox:false, random:false, seCorr:false} : v})]
                    })
                }} />
              </td>,
              /* Always Include */
              <td class="center">
                <input type="checkbox" checked={variable.always}  disabled={!variable.include} onChange={(e) => {
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {...v, always:!v.always} : v})]
                    })
                }} />
              </td>,
              /* BoxCox */
              <td class="center">
                <input type="checkbox" checked={variable.boxcox} disabled={!variable.include || boxcoxDisabled} onChange={(e) => {
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {...v, boxcox:!v.boxcox, seCorr:false} : v})]
                    })
                }} />
              </td>,
              /* Random */
              <td class="center">
                <input type="checkbox" checked={variable.random} disabled={!variable.include || !isMixed || randomDisabled} onChange={(e) => {
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {...v, random:!v.random, seCorr:false} : v})]
                    })
                }} />
              </td>,
              /* Correlated */
              <td class="center">
                <input type="checkbox" checked={variable.seCorr} disabled={!variable.random || corrDisabled} onChange={(e) => {
                    setAppData({...appData,
                      variables: [...appData.variables.map((v, j) =>  {return i===j ? {...v, boxcox:false, seCorr:!v.seCorr} : v})]
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

export function Metaheristic({appData, setAppData}) {
  /*
    component to change the metahuristic and choose an algoritum for the search
    
    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  return (
    <div class="section_mt_24">
      <div class="label">Algorithm</div>
      <div class="algorithm_buttons">
        <div 
          key="Simulated Annealing" 
          onClick={() => setAppData({...appData, algorithm:"Simulated Annealing"})}
          class={`model_button ${appData.algorithm==="Simulated Annealing" ? "model_button_select" : ""}`}
        >Simulated Annealing</div>
        <div 
          key="Harmomny Search" 
          onClick={() => setAppData({...appData, algorithm:"Harmony Search"})}
          class={`model_button ${appData.algorithm==="Harmony Search" ? "model_button_select" : ""}`}
        >Harmony Search</div>
      </div>
    </div>
  )
}

export function SearchParams({appData, setAppData}) {
  /*
    component to change the search parameters
    
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

      <Grid cols={3}>
        <SliderField label="Steps" value={appData.steps} min={5} max={100} onChange={sapp("steps")} hint="Annealing steps" />
        <SliderField label="Iterations" value={appData.iterations} min={5} max={200} onChange={sapp("iterations")} hint="Candidates per step" />
        <SliderField label="Draws (R)" value={appData.draws} min={100} max={5000} step={100} onChange={sapp("draws")} hint="Simulation draws" />
      </Grid>

      <div style={{height:"20px"}} />

      <Grid cols={2}>
        <SelectField label="Criterion" value={appData.criterion} onChange={sapp("criterion")}
          options={["bic","aic","ll"]} hint="Model selection criterion" />
      </Grid>

      <div class="tolerance_select_box">
        <SelectField
          label="Tolerance (gtol)" 
          value={appData.gtol} 
          onChange={sapp("gtol")} 
          options={["1e-3","1e-4","1e-5","1e-6"]} 
          hint="Gradient tolerance" 
        />
      </div>
    </div>
  )
}

function generatePy(appData) {
  /*
    Tempory legacy functionality for exporting python files directly

    appData -> the full app state object from App.jsx
  */
  const included  = appData.variables.filter(v => v.include);
  const always    = included.filter(v => v.always);

  const varList    = included.map(v => `'${v.name}'`).join(", ");
  const psAsvars   = always.filter(v => !v.random).map(v => `'${v.name}'`).join(", ");
  const psCorvars  = always.filter(v => v.random && v.seCorr).map(v => `'${v.name}'`).join(", ");
  const psBcvars   = always.filter(v => v.random && v.boxcox).map(v => `'${v.name}'`).join(", ");
  const randList   = included.filter(v => v.random).map(v => `'${v.name}'`).join(", ");
  const corrList   = included.filter(v => v.seCorr).map(v => `'${v.name}'`).join(", ");
  const bcList     = included.filter(v => v.boxcox).map(v => `'${v.name}'`).join(", ");
  const fixList    = included.filter(v => !v.random).map(v => `'${v.name}'`).join(", ");
  const callFn     = appData.algorithm === "Simulated Annealing" ? "call_siman" : "call_harmony";

  return `import pandas as pd
import sys, numpy as np, random, os
sys.path.insert(0, "/home/tacomora/SearchLibrium")
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["TF_DETERMINISM_OPS"] = "1"

from SearchLibrium.search import Parameters
from SearchLibrium.call_meta import ${callFn}

df = pd.read_csv('${appData.file.name || "data.csv"}')

choice_id = df['${appData.choiceId}']
ind_id    = df['${appData.indId}']
choice_var = df['${appData.choiceVar}']
alt_var    = df['${appData.altVar}']
choice_set = [${appData.choiceSet.split(",").map(s => `'${s.trim()}'`).join(",")}]

varnames   = [${varList}]
asvarnames = varnames
isvarnames = []
base_alt   = ${appData.baseAlt === "None" ? "None" : `'${appData.baseAlt}'`}

R   = ${appData.draws}
gTol = ${appData.gtol}
steps  = ${appData.steps}
iterations = ${appData.iterations}
name   = "${appData.runName}"

np.random.seed(50)
random.seed(50)

criterions = [['${appData.criterion}', -1]]
parameters = Parameters(
    criterions=criterions,
    df=df,
    choice_set=choice_set,
    choice_id=choice_id,
    alt_var=alt_var,
    varnames=varnames,
    isvarnames=isvarnames,
    asvarnames=asvarnames,
    choices=choice_var,
    ind_id=ind_id,
    base_alt=base_alt,
    allow_random=${appData.allowRandom ? "True" : "False"},
    allow_corvars=${appData.allowCorr ? "True" : "False"},
    allow_bcvars=${appData.allowBC ? "True" : "False"},
    n_draws=R,
    gtol=gTol,
    models=["${appData.model}"],
    fit_intercept=${appData.fitIntercept ? "True" : "False"},
    avail=${appData.avVar !== "None" ? `df['${appData.avVar}']` : "None"},
    verbose=False,
${psAsvars ? `    ps_randvars=[${psAsvars}],\n` : ""}${psCorvars ? `    ps_corvars=[${psCorvars}],\n` : ""}${psBcvars ? `    ps_bcvars=[${psBcvars}],\n` : ""}
${randList ? `    randvars=[${randList}],\n` : ""}${corrList ? `    corvars=[${corrList}],\n` : ""}${bcList ? `    bcvars=[${bcList}],\n` : ""}${fixList ? `    fixvars=[${fixList}],\n` : ""})
init_sol = None
search = ${callFn}(parameters, init_sol, ctrl=(1000, 1, steps, iterations), id_num=name)
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
    a.download = `${appData.runName}.py`;
    a.click();
  };

  return <button onClick={handleExport} class="export_button">↓ Export .py</button>
}

async function handleRun(appData, setAppData, csv, saRun, setSaRun) {
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
  const socket = new WebSocket(`${proto}//${window.location.host}/ws/run_search`)

  console.log(csv)
  // Connection opened
  socket.addEventListener("open", event => {
    setSaRun({...saRun, status: "running"})
    socket.send(JSON.stringify(appData))
    socket.send(csv) // sends as blob
  });

  // Listen for messages
  socket.addEventListener("message", event => {
    const msg = JSON.parse(event.data)
    setSaRun({...saRun, 
      status: msg.status ?? "done",
      total:  msg.total ?? 0,
      complete: msg.complete ?? 0,
    })
  });
}

export function RunButton({appData, setAppData, csv, saRun, setSaRun}) {
  /*
    Button that runs the model

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    csv         -> state to store the csv file
  */
  return (
    <button onClick={() => handleRun(appData, setAppData, csv, saRun, setSaRun)} class="run_button">▶ Run Search</button>
  )
}
