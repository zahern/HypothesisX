import { useEffect, useRef, useState } from "react";
import Plot from "./Plot.jsx";
import { COLORS, usePlotTheme } from "./helpers.js";

/*
  Model Comparison: a ranked table of the Top solutions plus three grouped bar
  charts (|LogLik|, BIC, AIC). Ported from the original summary IIFE.
*/
export default function Summary({ data }) {
  const PLOT_THEME = usePlotTheme();
  const sols = data.solutions;
  const best = Math.min(...sols.map((s) => s.bic));

  // All three charts in chart-grid-3 are equal width; observe the first to
  // size labels. Per-bar width ≈ inner * (1 - bargap) / nBars because every
  // trace shares one x-category in barmode:"group".
  const gridRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(0);
  useEffect(() => {
    const el = gridRef.current && gridRef.current.firstElementChild;
    if (!el) return;
    if (typeof ResizeObserver === "undefined") {
      setChartWidth(el.clientWidth);
      return;
    }
    const ro = new ResizeObserver(([entry]) => setChartWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [sols.length]);

  const nBars = sols.length || 1;
  const innerWidth = Math.max(0, chartWidth - 85);
  const barWidth = (innerWidth * 0.8) / nBars;
  const fontSize = Math.max(9, Math.min(15, Math.round(barWidth * 0.22)));

  const convergenceCell = (s) => {
    if (s.gtol_ok !== false && s.ftol_ok !== false) {
      return <span className="badge b-ok">✓ OK</span>;
    }
    const pts = [];
    if (s.gtol_ok === false && s.gtol_val !== null)
      pts.push(<span key="g" className="badge b-warn">gtol: {s.gtol_val.toExponential(2)}</span>);
    if (s.ftol_ok === false && s.ftol_val !== null)
      pts.push(<span key="f" className="badge b-warn">ftol: {s.ftol_val.toExponential(2)}</span>);
    return pts.length ? pts : <span className="badge b-warn">Near</span>;
  };

  const charts = [
    ["Log-Likelihood (|LL|)", sols.map((s) => (s.loglik ? Math.abs(s.loglik) : 0))],
    ["BIC", sols.map((s) => s.bic || 0)],
    ["AIC", sols.map((s) => s.aic || 0)],
  ];

  return (
    <div className="section">
      <h1>Model Comparison</h1>

      <div className="card">
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Rank</th><th>BIC</th><th>AIC</th><th>LogLik</th><th>Adj.ρ²</th>
                <th>#Fixed</th><th>#Random</th><th>#Corvars</th><th>Convergence</th>
              </tr>
            </thead>
            <tbody>
              {sols.map((s, i) => (
                <tr key={i}>
                  <td>
                    <b>Top {i + 1}</b>
                  </td>
                  <td><b>{Math.round(s.bic)}</b></td>
                  <td>{s.aic ? Math.round(s.aic) : "-"}</td>
                  <td>{s.loglik ? Math.round(s.loglik) : "-"}</td>
                  <td>{s.adjlik ? s.adjlik.toFixed(3) : "-"}</td>
                  <td>{s.fixed.length}</td>
                  <td>{s.random.length}</td>
                  <td>{s.corrvars.length}</td>
                  <td>{convergenceCell(s)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3>Goodness-of-Fit Comparison</h3>
        <div className="chart-grid-3" ref={gridRef}>
          {charts.map(([label, vals]) => {
            const yMin = Math.floor(Math.min(...vals) * 0.9);
            const yMax = Math.ceil(Math.max(...vals) * 1.1);
            return (
              <Plot
                key={label}
                style={{ height: 450 }}
                data={sols.map((s, i) => ({
                  name: "Top " + (i + 1), x: [label], y: [vals[i] - yMin],
                  base: yMin,
                  type: "bar", marker: { color: COLORS[i] },
                  text: [String(Math.round(vals[i]))],
                  textposition: "inside",
                  insidetextanchor: "start",
                  textfont: { color: PLOT_THEME.titleColor, size: fontSize },
                  hovertemplate: `Top ${i + 1}: ${Math.round(vals[i])}<extra></extra>`,
                }))}
                layout={{
                  ...PLOT_THEME,
                  barmode: "group", margin: { t: 20, b: 40, l: 65, r: 20 },
                  title: { text: label, font: { size: 13, color: PLOT_THEME.titleColor } },
                  legend: { ...PLOT_THEME.legend, orientation: "h", y: -0.25 },
                  xaxis: { ...PLOT_THEME.xaxis },
                  yaxis: { ...PLOT_THEME.yaxis, title: label, range: [yMin, yMax] },
                  uniformtext: { mode: "hide", minsize: fontSize },
                }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
