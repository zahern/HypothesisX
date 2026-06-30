import { useState } from "react";

import InputApp from "./input/App.jsx";
import OutputApp from "./output/App.jsx";
import Landing from "./Landing.jsx";

/*
  Top-level wrapper. First visit shows the Landing chooser; once the user
  picks a side, the relevant sub-app renders. Each sub-app owns its own theme
  state internally, matching the original input_gui / output_gui projects.
*/
export default function App() {
  const [mode, setMode] = useState("landing");

  if (mode === "input")  return <InputApp />;
  if (mode === "output") return <OutputApp />;
  return <Landing setMode={setMode} />;
}
