import { useEffect, useState } from "react";

import InputApp from "./input/App.jsx";
import OutputApp from "./output/App.jsx";
import Landing from "./Landing.jsx";

/*
  Top-level wrapper. First visit shows the Landing chooser; once the user
  picks a side, mode is set to "input" or "output" and the relevant sub-app
  renders. Mode and theme are owned here so they survive switching sides; each
  sub-app receives them as props and forwards them to its Banner, which
  renders the Input/Output and Dark/Light toggles.
*/
export default function App() {
  // Always start on the landing page — the chooser is shown on every fresh
  // load regardless of which sub-app the user picked last time.
  const [mode, setMode] = useState("landing");
  const [theme, setTheme] = useState(
    () => (localStorage.getItem("unified-gui-theme") === "light" ? "light" : "dark")
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("unified-gui-theme", theme);
  }, [theme]);

  if (mode === "input")  return <InputApp  theme={theme} setTheme={setTheme} />;
  if (mode === "output") return <OutputApp theme={theme} setTheme={setTheme} />;
  return <Landing setMode={setMode} theme={theme} setTheme={setTheme} />;
}
