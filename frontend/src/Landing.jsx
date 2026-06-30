/*
  Entry-point chooser. Two big cards: start a new estimation/search (opens the
  Input sub-app) or view a stored run (opens the Output dashboard). Picking one
  sets `mode` on the top-level App, which then renders the appropriate sub-app.
*/
export default function Landing({ setMode, theme, setTheme }) {
  return (
    <div className="landing-root">
      <div className="landing-banner">
        <div className="landing-logo">λ</div>
        <div>
          <div className="landing-title">DCM Studio</div>
          <div className="landing-sub">DISCRETE CHOICE MODELLING SUITE</div>
        </div>
        <div className="landing-spacer" />
        <div className="banner-mode-group" role="group" aria-label="Theme">
          <div className={`mode-button ${theme === "dark" ? "active" : ""}`}
               onClick={() => setTheme("dark")}>Dark</div>
          <div className={`mode-button ${theme === "light" ? "active" : ""}`}
               onClick={() => setTheme("light")}>Light</div>
        </div>
      </div>

      <div className="landing-body">
        <h1 className="landing-h1">What would you like to do?</h1>

        <div className="landing-grid">
          <div className="landing-card" onClick={() => setMode("input")}>
            <div className="landing-card-icon">▶</div>
            <div className="landing-card-title">Run a new search / estimation</div>
            <div className="landing-card-desc">
              Upload data, configure a model, and run either a single estimation
              or a metaheuristic search over model specifications.
            </div>
            <div className="landing-card-action">Go to Input →</div>
          </div>

          <div className="landing-card" onClick={() => setMode("output")}>
            <div className="landing-card-icon">📊</div>
            <div className="landing-card-title">View a stored run</div>
            <div className="landing-card-desc">
              Browse results from previously completed runs — convergence,
              coefficients, distributions, correlations, and predicted shares.
            </div>
            <div className="landing-card-action">Go to Output →</div>
          </div>
        </div>
      </div>
    </div>
  );
}
