# Setup
Windows: <br />
run setup.bat <br />
run main.py with python <br />

Linux:
```bash
cd frontend
npm install
npm run build
cd ..
python -m pip install requirements.txt
python main.py
```

This should all run a server at localhost:7823, otherwise the port will be indicated in the startup message.

# Notes
## General
If you have been given this codebase to work on I'll set out some notes about the general implementation of things. <br />

The general setup of the codebase is a backend of FastAPI in python and a frontend with react. If you don't want to worry about the config of anything and just want to add or modify code, you can recompile the frontend with the command "npm run build" run in the frontend directory. <br />

In general, I've preferred to keep things as modular as possible. There is some interdependence of variables between functions because they're all working on the same "appData" object, but other than in the VariableOptions and ModelButtons functions and the "variables" object stored in appData, most variables are only used at their inputs and at the model running.

## File Structure
As for the setup of the file structure it's standard for react. If you don't know much about react then you only need to know that the App.jsx in the frontend/src/ directory is the main file. This file calls files in the Components subdirectory which contains relatively intuitively named files and functions. If you need to investigate where a function is coming from, it'll be in the imports at the top of a file, declared in the file itself, or a built-in javascript function. <br />

A rough layout of the repository:
```
input_gui/
├── main.py                  FastAPI backend. Serves the built frontend from
│                            frontend/dist and exposes the /ws/run_estimation
│                            and /ws/run_search websocket endpoints (currently
│                            dummy implementations that just echo progress).
├── setup.bat                Windows convenience script (installs npm deps and
│                            builds the frontend so main.py can serve it).
├── requirements.txt         Python packages needed to run main.py.
├── README.md                This file.
└── frontend/                The React app.
    ├── index.html           Vite entry HTML; mostly a placeholder that loads main.jsx.
    ├── package.json         npm config / scripts (notably `npm run build`).
    ├── vite.config.js       Vite build config.
    ├── dist/                Build output from `npm run build`. This is what
    │                        FastAPI actually serves — you must rebuild after
    │                        any change to the frontend source.
    └── src/
        ├── main.jsx         React entry point; mounts <App/> into index.html.
        ├── App.jsx          Top-level component. Owns the big `appData` state
        │                    object described in the "General" section, plus
        │                    the csv state, and swaps between Estimation and
        │                    Search based on appData.mode.
        ├── All.css          All styling for the app. Color variables and
        │                    theme tokens live at the top.
        └── Components/
            ├── Banner.jsx                    Top banner (title, theme toggle,
            │                                 mode switch).
            ├── Sidebar.jsx                   Left sidebar (step navigation and
            │                                 space for an info box).
            ├── Estimation.jsx                Estimation mode page — lays out
            │                                 the steps for the estimation flow.
            ├── Estimation-Sub-Components.jsx Step components used only by
            │                                 Estimation.jsx (options panels,
            │                                 run button, results, etc.).
            ├── Search.jsx                    Search mode page — analogous to
            │                                 Estimation.jsx but for the search
            │                                 workflow.
            ├── Search-Sub-Components.jsx     Step components used only by
            │                                 Search.jsx.
            ├── Sub-Components.jsx            Shared components used by both
            │                                 Estimation and Search (file upload,
            │                                 VariableOptions, ModelButtons, and
            │                                 other reusable pieces).
            └── VarTable.jsx                  The variables table shown in the
                                              variable-selection step.
```

## Style
One of the choices I made was to try and detangle most of the style information in the frontend into a single css file. This worked out well for the style switching, but it can be daunting to find style information of an object. To find the where the style information of an object is, find its class either by inspecting it in a browser or looking at the source files, and search for that class in All.css.

## WebSockets
Running a model (both estimation and search) talks to the backend over a WebSocket rather than a normal HTTP request. I went this way because a run is long-lived and needs to stream progress back to the UI as it goes, which a single request/response doesn't do nicely.

The backend (`main.py`) exposes two WebSocket endpoints:
- `/ws/run_estimation` — handled by `handle_estimation_ws`
- `/ws/run_search` — handled by `handle_search_ws`

Both are dummy implementations for now. They accept the connection, read the data the frontend sends, then stream back fake progress/result messages so the UI can be built against a realistic message flow. When a real estimation/search engine is wired in, these handlers are where it goes.

The message flow is the same for both:
1. The frontend opens the socket (in the `handleRun` function inside `Estimation-Sub-Components.jsx` / `Search-Sub-Components.jsx`). The `ws:`/`wss:` scheme is picked automatically from the page protocol, and it connects to `window.location.host`, so it works both behind the FastAPI server and the Vite dev server without hard-coding a URL.
2. On `open`, the frontend sends two messages: first the whole `appData` object as JSON (`socket.send(JSON.stringify(appData))`), then the raw CSV `File` as a binary blob (`socket.send(csv)`). The backend reads these in the same order with `receive_json()` then `receive_bytes()` — so the order matters.
3. The backend streams back JSON messages, each with a `status` field. Estimation sends `{"status":"running"}` then a final `{"status":"done", ll, bic, aic}`. Search sends a stream of `{"status":"running", "total", "complete"}` progress ticks then `{"status":"done"}`. The frontend's `message` listener parses each one and pushes it into the run state (`estRun`/`saRun`), which drives the spinner/progress UI.
4. The backend closes the socket when the run finishes.

Some caveats to be aware of if you extend this:
- There's no verification or error handling on either side yet. The backend assumes both messages arrive in order and the socket stays connected; the frontend assumes every message is valid JSON. Real error/disconnect handling will need adding.
- The shape of `appData` sent over the wire will likely change once the real backend defines what it actually needs.
- The frontend sets up the socket with raw `addEventListener` calls inside `handleRun`. If this gets more complex, the `react-use-websocket` package is worth a look (noted in the code comments too) for tying the socket lifecycle to React more reliably.

## Other
if more data is wanted at a glance, the sidebar has space for an info box


# TODO
- add an animation to csv upload so when the file is eventually uploaded to a server it can take time <br />
- I don't love how the unique values in each column are stored for the choiceSet variable, it precomputes everything which could be bad
- The Run buttons for search and estimation don't have full functionality so that could be implemented and the "generate python" buttons could be removed 
- More warnings as to why things aren't working
