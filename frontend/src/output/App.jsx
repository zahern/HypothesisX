import { useState, useEffect } from "react";

import Banner from "./Components/Banner.jsx";
import Sidebar from "./Components/Sidebar.jsx";
import Summary from "./Components/Summary.jsx";
import Convergence from "./Components/Convergence.jsx";
import Distributions from "./Components/Distributions.jsx";
import Correlations from "./Components/Correlations.jsx";
import Shares from "./Components/Shares.jsx";
import Coefficients from "./Components/Coefficients.jsx";

/*
  Top-level app. Mirrors the structure of the sibling DCM Studio project: a
  banner across the top, a dark sidebar of navigation items, and a main panel
  that swaps section components. All dashboard data comes from the FastAPI
  backend (/api/runs, /api/dashboard) instead of being baked into the HTML.
*/
export default function App() {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState(null);
  const [data, setData] = useState(null);
  const [section, setSection] = useState("summary");
  const [error, setError] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [theme, setTheme] = useState(() =>
    localStorage.getItem("dcm-studio-theme") === "light" ? "light" : "dark"
  );

  // Apply theme to <html> so the CSS variable overrides in All.css activate,
  // and persist the choice so it sticks across reloads.
  useEffect(() => {
    document.documentElement.setAttribute("dcm-studio-theme", theme);
    localStorage.setItem("dcm-studio-theme", theme);
  }, [theme]);

  // Discover available runs once.
  useEffect(() => {
    fetch("/api/runs")
      .then((r) => r.json())
      .then((d) => {
        setRuns(d.runs || []);
        if (d.runs && d.runs.length) setRunId(d.runs[0]);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // (Re)load the dashboard payload whenever the selected run changes.
  useEffect(() => {
    if (!runId) return;
    setData(null);
    fetch(`/api/dashboard?run_id=${encodeURIComponent(runId)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e)));
  }, [runId]);

  // Choose the starting section when data is updated
  useEffect(() => {
    setSection(data?.solutions.length > 1 ? "summary" : "coefficients")
  }, [data])

  // If the active section becomes unavailable for this run (e.g. no random
  // params), fall back to the summary.
  useEffect(() => {
    if (!data) return;
    const ok = {
      summary: true,
      convergence: data.flags.hasConvergence,
      distributions: data.flags.hasRandom,
      correlations: data.flags.hasCorvars,
      shares: true,
      coefficients: true,
    };
    if (!ok[section]) setSection("summary");
  }, [data, section]);

  // On narrow screens the sidebar is an off-canvas drawer; close it whenever
  // the user picks a new section so they aren't left staring at the overlay.
  useEffect(() => { setDrawerOpen(false); }, [section]);

  const renderSection = () => {
    if (!data) return null;
    switch (section) {
      case "summary": return <Summary data={data} />;
      case "convergence": return <Convergence data={data} />;
      case "distributions": return <Distributions data={data} runId={runId} />;
      case "correlations": return <Correlations data={data} />;
      case "shares": return <Shares data={data} runId={runId} />;
      case "coefficients": return <Coefficients data={data} />;
      default: return <Summary data={data} />;
    }
  };

  return (
    <>
      <Banner runs={runs} runId={runId} setRunId={setRunId}
              theme={theme} setTheme={setTheme}
              onMenuToggle={() => setDrawerOpen((o) => !o)} />
      <div className="body-row">
        <Sidebar data={data} section={section} setSection={setSection}
                 drawerOpen={drawerOpen} onClose={() => setDrawerOpen(false)} />
        {drawerOpen && <div className="drawer-backdrop"
                            onClick={() => setDrawerOpen(false)} />}
        <div id="main">
          {error && <div className="card error-msg">Error: {error}</div>}
          {!data && !error && <div className="loading">Loading dashboard…</div>}
          {renderSection()}
        </div>
      </div>
    </>
  );
}
