import { useRef, useState } from "react";
import Plot from "./Plot.jsx";
import { sbgCls, usePlotTheme } from "./helpers.js";
import { captureCardAsPng } from "./screenshot.js";
import MiniSummary from "./MiniSummary.jsx"

/*
  Parameter Distributions: a table with one row per random variable and one
  column per Top solution. Each present cell shows the fitted density (split
  red/green at zero) plus a negative/positive share bar. Ported from renderDist.
*/

// Build the red (x<=0) / green (x>0) fill traces for one density curve.
function distTraces(p) {
  const xs = p.xs, ys = p.ys;
  const nx = [], ny = [], px = [], py = [];
  xs.forEach((x, i) => {
    if (x <= 0) { nx.push(x); ny.push(ys[i]); }
    else { px.push(x); py.push(ys[i]); }
  });
  if (nx.length && px.length) {
    nx.push(0); ny.push(ny[ny.length - 1]);
    px.unshift(0); py.unshift(py[0]);
  }
  const trs = [];
  if (nx.length) trs.push({ x: nx, y: ny, fill: "tozeroy", mode: "lines", line: { color: "#f87171", width: 2 }, fillcolor: "rgba(248,113,113,0.22)", showlegend: false });
  if (px.length) trs.push({ x: px, y: py, fill: "tozeroy", mode: "lines", line: { color: "#4ade80", width: 2 }, fillcolor: "rgba(74,222,128,0.22)", showlegend: false });
  return trs;
}

// Header / variable-cell appearance lives in All.css (.dist-head, .dist-var-cell)
// so they override the global <th>/<td> resets in one place.

export default function Distributions({ data, runId }) {
  const PLOT_THEME = usePlotTheme();
  const DIST_LAYOUT = {
    ...PLOT_THEME,
    paper_bgcolor: "transparent",
    margin: { t: 4, b: 24, l: 24, r: 4 },
    xaxis: { ...PLOT_THEME.xaxis, zeroline: true, zerolinecolor: PLOT_THEME.font.color, zerolinewidth: 1.5 },
    yaxis: { ...PLOT_THEME.yaxis, showticklabels: false },
  };

  const distData = data.distData;
  const sols = data.solutions;

  // Union of all variables, in solution-array order.
  const seen = new Set();
  const allVars = [];
  sols.forEach((s) => {
    (distData[s.rank] || []).forEach((p) => {
      if (!seen.has(p.var)) { seen.add(p.var); allVars.push(p.var); }
    });
  });

  const cardRef = useRef(null);
  const [saving, setSaving] = useState(false);

  const handleExport = async () => {
    setSaving(true);
    try {
      await captureCardAsPng(cardRef.current, {
        filename: `parameter_distributions_${runId || "run"}`,
        expandScrollable: true,
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="section">
      <h1>Parameter Distributions</h1>

      {data?.solutions.length === 1 && (
        <div class="card">
          <MiniSummary s={data?.solutions[0]} />
        </div>
      )}

      <div className="card" ref={cardRef}>
        <div className="section-toolbar no-export">
          <div className="btn-group">
            <button onClick={handleExport} disabled={!allVars.length || saving}>
              {saving ? "Saving…" : "Save as PNG"}
            </button>
          </div>
        </div>
        {allVars.length === 0 ? (
          <p className="empty">No random parameters.</p>
        ) : (
          <div className="table-scroll">
            <table id="dist-container">
              <thead>
                <tr>
                  <th />
                  {sols.map((s, i) => (
                    <th key={`h-${i}`} className="dist-head">
                      Top {i + 1}<br />
                      <span className="dist-bic-sub">BIC={Math.round(s.bic)}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {allVars.map((varName) => (
                  <tr key={`v-${varName}`}>
                    <td className="dist-var-cell">{varName}</td>
                    {sols.map((sol, i) => {
                      const p = (distData[sol.rank] || []).find((x) => x.var === varName);
                      if (!p) {
                        const fixed = sol && sol.fixed && sol.fixed.find((f) => f.var === varName);
                        if (fixed) {
                          return (
                            <td key={i}>
                              <div className="dist-card dist-card-tall">
                                <div className="dist-title">
                                  <span className="dmeta">[fixed]</span>
                                  <span className={`sbg ${sbgCls(fixed.sig)}`} title="coeff sig">β {fixed.sig}</span>
                                </div>
                                <div className="dist-fixed-val">
                                  β = {fixed.coeff.toFixed(4)}
                                </div>
                              </div>
                            </td>
                          );
                        }
                        return (
                          <td key={i}>
                            <div className="dist-empty-card">
                              <p>—</p>
                            </div>
                          </td>
                        );
                      }
                      return (
                        <td key={i}>
                          <div className="dist-card">
                            <div className="dist-title">
                              <span className="dmeta">[{p.dist}] μ={p.mean.toFixed(3)} σ={Math.abs(p.sd).toFixed(3)}</span>
                              <span className={`sbg ${sbgCls(p.sig)}`} title="mean sig">μ {p.sig}</span>
                              <span className={`sbg ${sbgCls(p.sig_sd || "ns")}`} title="sd sig">σ {p.sig_sd || "-"}</span>
                            </div>
                            <Plot style={{ height: 180 }} data={distTraces(p)} layout={DIST_LAYOUT} />
                            <div className="pct-row">
                              <div className="pct-bar">
                                <div className="pct-neg" style={{ width: `${p.pct_neg}%` }} />
                                <div className="pct-pos" style={{ width: `${p.pct_pos}%` }} />
                              </div>
                              <span className="pct-lbl">Neg:{p.pct_neg}% Pos:{p.pct_pos}%</span>
                            </div>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
