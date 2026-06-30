import { useEffect, useState } from "react";

import InputApp from "./input/App.jsx";
import OutputApp from "./output/App.jsx";
import Landing from "./Landing.jsx";

/*
  Top-level wrapper. `mode` is mirrored in the URL path so the Input and
  Output sub-apps are reachable directly at /input and /output. The Landing
  chooser is shown at /. Browser back/forward (popstate) keeps mode in sync.
*/
const modeFromPath = (path) => {
  if (path === "/input")  return "input";
  if (path === "/output") return "output";
  return "landing";
};

const pathFromMode = (mode) => {
  if (mode === "input")  return "/input";
  if (mode === "output") return "/output";
  return "/";
};

export default function App() {
  const [mode, setModeState] = useState(() => modeFromPath(window.location.pathname));

  useEffect(() => {
    const onPop = () => setModeState(modeFromPath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const setMode = (next) => {
    const nextPath = pathFromMode(next);
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
    }
    setModeState(next);
  };

  let view;
  if (mode === "input")       view = <InputApp />;
  else if (mode === "output") view = <OutputApp />;
  else                        view = <Landing setMode={setMode} />;

  return (
    <>
      {view}
      <div class="app_copyright">GUI © 2026 Fernando Taco-Morales</div>
    </>
  );
}
