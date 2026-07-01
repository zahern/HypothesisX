import { useState } from "react";
import Plot from "./Plot.jsx";
import { usePlotTheme } from "./helpers.js";
import { SummaryTable, MiniSummaryTable } from "./SummaryTables.jsx"

/*
  Convergence section: search trajectory (objective per evaluation + running
  best) with an All / Accepted-only toggle, plus a per-solution optimizer
  convergence panel. The optimizer panel needs grad_history, which is only
  present when the dashboard is generated live from a siman object — parsing
  the result files alone leaves it empty, so it shows a placeholder.
*/
export default function Convergence({ data }) {
  const PLOT_THEME = usePlotTheme();
  const conv = data.convergence;
  const sols = data.solutions;
  const [mode, setMode] = useState("all");
  const [idx, setIdx] = useState(0);

  const it = conv.iterations, bc = conv.bics, ac = conv.accepted, bst = conv.best_bics;

  let xi, yi, ci;
  if (mode === "accepted") {
    xi = it.filter((_, i) => ac[i]);
    yi = bc.filter((_, i) => ac[i]);
    ci = yi.map(() => "#60a5fa");
  } else {
    xi = it; yi = bc;
    ci = bc.map((_, i) => (ac[i] ? "#60a5fa" : "#f87171"));
  }

  const sol = sols[idx];
  const hasOptim = sol && sol.grad_history && sol.grad_history.length > 0;

  let optimLL = null, optimGrad = null;
  if (hasOptim) {
    const evals = sol.grad_history.map((h) => h[0]);
    const ll = sol.grad_history.map((h) => h[1]);
    const grad = sol.grad_history.map((h) => h[2]);
    const llMono = ll.reduce((acc, v) => { acc.push(Math.min(v, acc.length ? acc[acc.length - 1] : v)); return acc; }, []);
    const gradMono = grad.reduce((acc, v) => { acc.push(Math.min(v, acc.length ? acc[acc.length - 1] : v)); return acc; }, []);
    optimLL = { evals, ll, llMono };
    optimGrad = { evals, grad, gradMono };
  }

  return (
    <div className="section">
      <h1>Convergence</h1>



      { data?.solutions.length > 1 && (
        <>
          <SummaryTable sols={sols} />
          
          <div className="card">
            <div className="controls">
              <label>Show:</label>
              <div className="btn-group">
                <button className={mode === "all" ? "active" : ""} onClick={() => setMode("all")}>All evaluations</button>
                <button className={mode === "accepted" ? "active" : ""} onClick={() => setMode("accepted")}>Accepted only</button>
              </div>
            </div>
            <Plot
              style={{ height: 440 }}
              data={[
                {
                  x: xi, y: yi, mode: "lines+markers", name: conv.objective + " per eval",
                  line: { color: "rgba(100,116,139,0.35)", width: 1 },
                  marker: { color: ci, size: 4, opacity: 0.75 }, type: "scatter",
                },
                {
                  x: it, y: bst, mode: "lines", name: "Best " + conv.objective,
                  line: { color: "#fbbf24", width: 2.5 }, type: "scatter",
                },
              ]}
              layout={{
                ...PLOT_THEME,
                margin: { t: 10, b: 50, l: 65, r: 20 },
                xaxis: { ...PLOT_THEME.xaxis, title: "Iteration" },
                yaxis: { ...PLOT_THEME.yaxis, title: conv.objective },
                legend: { ...PLOT_THEME.legend, orientation: "h", y: -0.2 },
              }}
            />
          </div>
        </>
      )}

      <div className="card">
        <div className="controls">
          <label>Optimizer Convergence:</label>
          { sols.length > 1 && (
            <select value={idx} onChange={(e) => setIdx(parseInt(e.target.value))}>
              {sols.map((s, i) => (
                <option key={i} value={i}>Top {i + 1}  (BIC={Math.round(s.bic)})</option>
              ))}
            </select>
          )}
        </div>

        <MiniSummaryTable s={sols[idx ?? 0]}/>

        {!hasOptim ? (
          <p className="empty">No optimizer history available.</p>
        ) : (
          <>
            <Plot
              style={{ height: 300, marginBottom: 12 }}
              data={[
                { x: optimLL.evals, y: optimLL.ll, mode: "lines", name: "LL", line: { color: "#c084fc", width: 1 } },
                { x: optimLL.evals, y: optimLL.llMono, mode: "lines", name: "LL Monotone", line: { color: "#fbbf24", width: 2 } },
              ]}
              layout={{
                ...PLOT_THEME,
                margin: { t: 10, b: 50, l: 65, r: 20 },
                xaxis: { ...PLOT_THEME.xaxis, title: "Evaluations" },
                yaxis: { ...PLOT_THEME.yaxis, title: "Log-Likelihood" },
                showlegend: false,
              }}
            />
            <Plot
              style={{ height: 300 }}
              data={[
                { x: optimGrad.evals, y: optimGrad.grad, mode: "lines", name: "Grad Norm", line: { color: "#c084fc", width: 1 } },
                { x: optimGrad.evals, y: optimGrad.gradMono, mode: "lines", name: "Grad Norm Monotone", line: { color: "#fbbf24", width: 2 } },
              ]}
              layout={{
                ...PLOT_THEME,
                margin: { t: 10, b: 50, l: 65, r: 20 },
                xaxis: { ...PLOT_THEME.xaxis, title: "Evaluations" },
                yaxis: { ...PLOT_THEME.yaxis, title: "Grad Norm (inf)", type: "log" },
                showlegend: false,
                shapes: [{
                  type: "line", x0: 0, x1: 1, xref: "paper",
                  y0: conv.gtol || 1e-5, y1: conv.gtol || 1e-5,
                  line: { color: "#f87171", dash: "dash", width: 1.5 },
                }],
              }}
            />
          </>
        )}
      </div>
    </div>
  );
}
