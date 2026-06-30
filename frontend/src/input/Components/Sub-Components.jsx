import "../All.css"

/* ------------------------------------ Helpers ------------------------------------ */
function initVar(vName) {
  /*
    returns the initial state of a variable object. Not all 
    values in this object are necesarally used in estimaiton 
    or search, but they are all used in at least one

    vName   -> string with the name of the new variable
  */
  return ({name:vName, dist:"none", include:true,
    random:false, boxcox:false, estCorr:false, seCorr:false, always:false})
}

// case-insensitive alphabetical sort of variable objects by .name
const byName = (a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase());

const SENTINEL = "Upload Data to Detect";
export function isDataModelComplete(appData) {
  /*
    True when a file is uploaded, all non-optional column selections
    on Data & Model have real (non-sentinel) values, and every selected
    role column is a distinct column.
    indId is only required for panel models (not MNL/RRM).
    avVar is optional; "None" means unset and is excluded from the
    distinctness check.

    appData -> state holding most data for the app
  */
  if (!appData.uploaded)                return false;
  if (appData.choiceId  === SENTINEL)   return false;
  if (appData.choiceVar === SENTINEL)   return false;
  if (appData.altVar    === SENTINEL)   return false;
  const needsIndId = appData.model !== "MNL" && appData.model !== "RRM";
  if (needsIndId && appData.indId === SENTINEL) return false;

  const roles = [appData.choiceId, appData.choiceVar, appData.altVar];
  if (needsIndId)               roles.push(appData.indId);
  if (appData.avVar !== "None") roles.push(appData.avVar);
  if (new Set(roles).size !== roles.length) return false;

  return true;
}

export function Grid({ cols, children, style }) {
  /*
    places children into a grid with cols columns

    cols      ->  number of columns
    children  ->  items to be placed into grid
    style     -> any extra style info
  */
  return <div class="grid" style={{ "--cols": cols, ...style }}>{children}</div>;
}

/* ------------------------------------ Fields ------------------------------------ */
export function SliderField({ label, value, min, max, step=1, onChange, hint }) {
  /*
    functino to place a slider for data entry
    
    label     -> string of text to be placed above the slider
    value     -> State containing the value of the slider
    min       -> number, lower bounds of the slider
    max       -> number, upper bounds of the slider
    step      -> number, incriment of the bar
    onChange  -> function to change value, when the bar is slid
    hint      -> string to be placed under the bar
  */
  return (
    <div>
      <div class="label">{label}</div>
      <div class="slider_row">
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          class="slider_range" />
        <input type="text" value={value} onChange={e => onChange(Number(e.target.value))}
          class="slider_number" />
      </div>
      {hint && <div class="hint">{hint}</div>}
    </div>
  );
}

export function Toggle({ label, active, onChange }) {
  /*
    function to place a toggle box, toggling a varibale between true and false

    label       -> string to be place to the right of the toggle field
    active      -> state containing a boolean represneting to togle of the field
    onChnage    -> function to change the active state on click
  */
  return (
    <div class="toggle" onClick={() => onChange(!active)}>
      <div class={`toggle_track ${active ? "toggle_track_active" : ""}`}>
        <div class={`toggle_thumb ${active ? "toggle_thumb_active" : ""}`} />
      </div>
      <span class={`toggle_label ${active ? "toggle_label_active" : ""}`}>{label}</span>
    </div>
  );
}

export function SelectField({ label, value, onChange, options, hint, disabled }) {
  /*
    generates a box with options to be selected from

    label     ->  label to be placed above a selection field
    value     ->  react useState variable 
    onChange  ->  react useState setter variable
    options   ->  list of strigns or objects like {value:"", label""} containing 
                  information for options for the selection box
    hint      ->  extra infromation that can be displayed underneth the box
    disabled  ->  boolean denoting if the box is disabled
  */
  
  // internal function so assume well formed inputs
  if (typeof options[0] === "string") {
    options = options.map((v, _) => {return {value: v, label: v}})
  }

  return (
    <div>
      {label && <div class="label">{label}</div>}
      <select
        value={value} 
        onChange={e => onChange(e.target.value)} 
        disabled={disabled}
        class={disabled ? "generic_input disabled_input" : "generic_input"}
      >
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
      {hint && <div class="hint">{hint}</div>}
    </div>
  );
}

/* ------------------------------------ Components ------------------------------------ */
export function SummaryBox({ rows }) {
  /*
    box that summarises much information, used at the final screen of both search and estimation

    rows -> array of strings or any data that can be displayed that gives the desired info
  */
  return (
    <div class="summary_box">
      {rows.map(([k,v]) => (
        <div key={k} class="summary_row">
          <span class="summary_key">{k}</span>
          <span class="summary_value">{v}</span>
        </div>
      ))}
    </div>
  );
}

export function Spinner({ label }) {
  /*
    loading spinner

    label -> string to label the spinner
  */
  return (
    <div class="spinner_wrap">
      <div class="spinner" />
      <div class="spinner_label">{label}</div>
    </div>
  );
}

export function DoneMsg({ label }) {
  /*
    prettely displays a done message at the end of a run

    label -> string to put in the message
  */
  return (
    <div class="done_msg">
      ✓ {label}
    </div>
  );
}

export function PageButtons({appData, setAppData, canNext, next, back}) {
  /*
    Function for the page navigation buttons at the bottom of pages

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    canNext     -> boolean representing if user can progress usefull to pass a state to
    next        -> boolean, if true it renders the next button, nothing if false
    back        -> boolean, if true it renders the back button, nothing if false
  */
  return (
    <div class="page_buttons_row">
      {back && <button
        onClick={() => setAppData({...appData, step: appData.step-1})}
        class={"back_button"}
      >
        ← Back
      </button>}
      {next && <button 
        onClick={() => setAppData({...appData, step: appData.step+1})}
        disabled={!canNext}
        class={`next_button ${!canNext ? "next_button_disabled" : ""}`}
      >
        Next →
      </button>}
    </div>
  )
}


/* ------------------------------------ Sections ------------------------------------ */
export function CSVUploadBox({appData, setAppData, csv, setCsv }) {
  /* 
    It was chosen to do the initial parseing of the columns here to keep the api RESTfull 
    I dont love keeping the csv sitting in the browser but sacrifices must be made 
    If csv filesizes get too big #TODO this can be tackled again 
    sub-funciton handleFile handles column parseing
    It also stores the file is csv

    Other than that this function creates the box for a user to upload their file 

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
    csv         -> state to store the csv file
    setCsv      -> function to set the state of csv
  */
  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();

    reader.onload = ({ target }) => {
      // parse file
      const lines = target.result.split('\n').filter(l => l.trim());
      const rawCols = lines[0].split(",").map(c => c.trim().replace(/"/g,""));
      // ignore blank-header columns (e.g. an unnamed index column)
      const cols = rawCols.filter(Boolean);

      // helpers
      const match = (patterns) => cols.find(c => patterns.some(p => c.toLowerCase() === p.toLowerCase())) || cols[0];
      
      // process stuff
      const tmpChoiceId  = match(["csn","chid","ch_id","choice_id","choiceid","choice_situation","cs"]);
      const tmpIndId     = match(["id","id_1","ind_id","individual_id","respondent_id","panelid","panel_id","resp_id"]);
      const tmpAltVar    = match(["scenario","alt","alternative","alt_var","altvar","alternative_id"]);
      const tmpChoiceVar = match(["choice_","choice","chosen","y","selected","choice_var"]);
      const tmpAvVar     = match(["av","avail","availability","available","av_var","alt_avail"]) === cols[0] ? "None" : match(["av","avail","availability","available","av_var","alt_avail"]);
      
      // pre-compute unique values per column so altVar changes can refresh choiceSet without re-parsing
      const rows = lines.slice(1).map(l => l.split(",").map(c => c.trim()));
      const columnUniques = Object.fromEntries(
        rawCols.flatMap((c, i) => c ? [[c, [...new Set(rows.map(r => r[i]).filter(Boolean))]]] : [])
      );
      const uniqueAlts = columnUniques[tmpAltVar] ?? [];

      // set new data for the app
      setAppData({
        ...appData,
        uploaded: true,
        ["file"]: {
          name: file.name,
          rows: lines.length - 1,
          cols: cols.length,
        },
        columns: ["None", ...cols],
        columnUniques,
        choiceId: tmpChoiceId,
        indId: tmpIndId,
        altVar: tmpAltVar,
        choiceVar: tmpChoiceVar,
        avVar: tmpAvVar,
        choiceSet: uniqueAlts.join(", "),
        variables: cols.filter(c => ![tmpChoiceId, tmpIndId, tmpAltVar, tmpChoiceVar, tmpAvVar].includes(c)).map(initVar).sort(byName),
      });
    };

    // save file temporarilly and run reader
    setCsv(file);
    reader.readAsText(file);
  }

  return (
    /* div for the data box, if a file is dragged onto it hadle it if its clicked it indirectly clicks the regular input for files that also handles this csv the same way */
    <div onDra gOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
      onClick={() => document.getElementById("csvInput").click()}
      class={`csv_upload ${appData.uploaded ? "csv_upload_uploaded" : ""}`}>
      {/* this input launces the file popup prompt when clicked, its invisible and clicked in proxy by prettyer elements */}
      <input id="csvInput" type="file" accept=".csv" class="csv_hidden_input"
        onChange={e => { console.log("file selected:", e.target.files[0]); handleFile(e.target.files[0]); }}
      />
      {appData.uploaded ? (
        <>
          <div class="csv_icon">✓</div>
          <div class="csv_filename">{appData.file.name}</div>
          <div class="csv_fileinfo">{`${appData.file.rows} rows · ${appData.file.cols} columns`}</div>
        </>
      ) : (
        <>
          <div class="csv_icon">↑</div>
          <div class="csv_hint">Drop CSV here or click to browse</div>
        </>
      )}
    </div>
  )
}


export function DataOptions({appData, setAppData}) {
  /*
    section to choose data options for the model

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  // helper to quickly generate functions to write to appData
  const sapp = (key) => {
    return (val) => setAppData({...appData, [key]: val})
  };

  const sApp = (key) => {
    return (c) => {
      // columns still claimed by the OTHER role selectors -> {role columns} REMOVE key
      const reserved = new Set(
        ["choiceId", "indId", "altVar", "choiceVar", "avVar"]
          .filter(k => k !== key)
          .map(k => appData[k])
      );

      const next = {...appData,
        variables: [
          ...appData.variables.map(v => v.name!==c ? v : {name:null}),
          initVar((appData[key]==="None" || reserved.has(appData[key])
          ) ? null : appData[key]), // skip None and columns still used by another role
        ].filter((v) => v.name!==null).sort(byName),
        [key]: c,
      };

      if (key === "altVar") {
        const uniques = appData.columnUniques?.[c] ?? [];
        next.choiceSet = uniques.join(", ");
        next.baseAlt = uniques.includes(appData.baseAlt) ? appData.baseAlt : "None";
      }

      setAppData(next);
    }
  }
  
  const columns = [...new Set([...appData.variables.map(v => v.name),
    appData.choiceId, appData.indId, appData.altVar,
    appData.choiceVar, appData.avVar
  ])]

  const altOptions = ["None", ...appData.choiceSet
    .split(",")
    .map(s => s.trim())
    .filter(Boolean)]

  return (
    <div> 
      <Grid cols={2}>
        <SelectField 
          label="Choice Situation ID" 
          value={appData.choiceId} 
          onChange={sApp("choiceId")}
          options={columns} 
          hint="Unique ID per choice task" 
          disabled={!appData.uploaded}
        />
        <SelectField 
          label="Individual (Panel) ID" 
          value={appData.indId} 
          onChange={sApp("indId")} 
          options={columns} 
          hint="Respondent identifier" 
          disabled={!appData.uploaded || appData.model === "MNL" || appData.model ==="RRM"}
        />
        <SelectField 
          label="Choice Variable" 
          value={appData.choiceVar} 
          onChange={sApp("choiceVar")} 
          options={columns} 
          hint="Column with observed choice (0/1)"
          disabled={!appData.uploaded}
        />
        <SelectField  
          label="Availability (optional)" 
          value={appData.avVar} 
          onChange={sApp("avVar")} 
          options={columns} 
          hint="Column with alternative availability (0/1)"
          disabled={!appData.uploaded}
        /> 
      </Grid>
      
      <div class="spacer_50" />

      <Grid cols={2}>
        <SelectField 
          label="Alternative Variable" 
          value={appData.altVar} 
          onChange={sApp("altVar")} 
          options={columns} 
          hint="Column identifying each alternative" 
          disabled={!appData.uploaded}
        />
        <SelectField 
          label="Base Alternative" 
          value={appData.baseAlt} 
          onChange={sapp("baseAlt")}
          options={altOptions}
          hint="Reference alternative for ASCs"
          disabled={!appData.uploaded}
        />
      </Grid>

      <div class="section_mt_20">
        <div class="label">Alternatives in Choice Set</div>
        <input 
          value={appData.choiceSet}
          onChange={e => {
            const choiceSet = e.target.value;
            const alts = choiceSet.split(",").map(s => s.trim()).filter(Boolean);
            setAppData({...appData,
              choiceSet,
              baseAlt: alts.includes(appData.baseAlt) ? appData.baseAlt : "None",
            });
          }}
          class={!appData.uploaded ? "generic_input disabled_input" : "generic_input"} 
          disabled={!appData.uploaded}
        />
        <div class="hint">Comma-separated list of alternative labels</div>
      </div>
    </div>
  )
}

export function ModelOptions({appData, setAppData}) {
  /*
    section to choose options for the model

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  // helper to quickly generate functions to write to appData
  const sapp = (key) => {
    return (val) => setAppData({...appData, [key]: val})
  };

  return (
    <div>
      <div class="intercept_block">
        <div class="label">Intercept</div>
        <div class="intercept_buttons">
          <div key="fitIntercept-yes" onClick={() => sapp("fitIntercept")(true)}
            class={appData.fitIntercept ? "model_button model_button_select" : "model_button"}>
            Include intercept
          </div>

          <div key="fitIntercept-no"  onClick={() => sapp("fitIntercept")(false)}
            class={appData.fitIntercept ? "model_button" : "model_button model_button_select"}>
            No intercept
          </div>
        </div>
      </div>
    </div>
  )
}

export function ModelSelect({appData, setAppData}) {
  /*
    section to choose between the 4 diferent models

    appData     -> state holding most data for the app
    setAppData  -> state setter for appData
  */
  return (
    <div class="section_mt_28">
      <div class="label">Model Type</div>

      <div class="model_choice_list">
        <div class="model_choice_row">
          <span class="text2 model_choice_label">Utility Maximization</span>
          <div class="model_choice_buttons">

            <div key="MNL" class={appData.model === "MNL" ? "model_button model_button_select" : "model_button"} onClick={() => {setAppData({...appData,
              model: "MNL",
              allowRandom: false,
              allowCorr: false,
              variables: appData.variables.map((v) => ({...v, dist:"none", random:false, estCorr:false, seCorr:false})),
            })}}>MNL</div>
            <div key="MXL" class={appData.model === "MXL" ? "model_button model_button_select" : "model_button"} onClick={() => {setAppData({...appData,
              model: "MXL",
              // Expected behaviour here is to set allow Random and Corr to true 
              // if switching from a non-Mixed model to a mixed one, or keep it 
              // the same when switching between mixed models
              allowRandom: (appData.model==="MXRRM" ? appData.allowRandom : true),
              allowCorr: (appData.model==="MXRRM" ? appData.allowCorr : true),
            })}}>MXL</div>
          </div>
        </div>


        <div class="model_choice_row">
          <span class="text2 model_choice_label">Regret Minimization</span>
          <div class="model_choice_buttons">
            <div key="RRM" class={appData.model === "RRM" ? "model_button model_button_select" : "model_button"} onClick={() => {setAppData({...appData,
              model: "RRM",
              allowRandom: false,
              allowCorr: false,
              variables: appData.variables.map((v) => ({...v, dist:"none", random:false, estCorr:false, seCorr:false})),
            })}}>RRM</div>
            <div key="MXRRM" class={appData.model === "MXRRM" ? "model_button model_button_select" : "model_button"} onClick={() => {setAppData({...appData,
              model: "MXRRM",
              // Expected behaviour here is to set allow Random and Corr to true 
              // if switching from a non-Mixed model to a mixed one, or keep it 
              // the same when switching between mixed models
              allowRandom: (appData.model==="MXL" ? appData.allowRandom : true),
              allowCorr: (appData.model==="MXL" ? appData.allowCorr : true),
            })}}>MXRRM</div>
          </div>
        </div>                 
      </div>  
      <div class="hint">MNL / RRM → no panel. MXL / MXRRM → panel + random params.</div>
    </div>
  )
}


