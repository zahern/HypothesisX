/*
  Top banner with the project identity, a Dark/Light theme switch, and a run
  selector. The run dropdown is populated from /api/runs; switching it reloads
  the dashboard in App.
*/
export default function Banner({ runs, runId, setRunId, theme, setTheme, onMenuToggle }) {
  return (
    <div className="banner">
      <button className="banner-menu" onClick={onMenuToggle} aria-label="Open navigation">☰</button>
      <a className="banner-home" href="/" aria-label="Go to dashboard home">
        <div className="banner-logo">λ</div>
        <div>
          <div className="banner-title">RESULTS Dashboard</div>
          <div className="banner-sub">DISCRETE CHOICE MODEL RESULTS</div>
        </div>
      </a>
      <div className="banner-spacer" />

      <div className="banner-mode-group" role="group" aria-label="Theme">
        <div className={`mode-button ${theme === "dark" ? "active" : ""}`}
             onClick={() => setTheme("dark")}>Dark</div>
        <div className={`mode-button ${theme === "light" ? "active" : ""}`}
             onClick={() => setTheme("light")}>Light</div>
      </div>

      <span className="banner-run-label">RUN</span>
      <select value={runId || ""} onChange={(e) => setRunId(e.target.value)}>
        {runs.length === 0 && <option value="">No runs found</option>}
        {runs.map((r) => (
          <option key={r} value={r}>{r}</option>
        ))}
      </select>
    </div>
  );
}
