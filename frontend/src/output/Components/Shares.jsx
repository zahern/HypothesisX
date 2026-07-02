import { useEffect, useRef, useState } from "react";
import Plot from "./Plot.jsx";
import { COLORS, usePlotTheme } from "./helpers.js";
import { captureCardAsPng } from "./screenshot.js";
import { SummaryTable, MiniSummaryTable } from "./SummaryTables.jsx"

/*
  Predicted vs Observed Shares: one chart per alternative. Each chart shows
  every Top solution's predicted share for that alternative as a coloured bar
  (x-axis = solution rank) against a single horizontal dotted line at the
  observed share. Each chart's y-axis min sits ~10% below the lowest value
  drawn so small inter-solution differences stay visible. Ported from
  renderShares.
*/
export default function Shares({ data, runId }) {
  const PLOT_THEME = usePlotTheme();
  const sols = data.solutions;
  const ALT_LBL = data.altLabels;
  const obs = sols[0] && sols[0].observed.length ? sols[0].observed : null;

  const cardRef = useRef(null);
  const [saving, setSaving] = useState(false);

  const handleExport = async () => {
    setSaving(true);
    try {
      await captureCardAsPng(cardRef.current, {
        filename: `shares_${runId || "run"}`,
      });
    } finally {
      setSaving(false);
    }
  };

  const solsWithIdx = sols
    .map((s, i) => ({ s, i }))
    .filter(({ s }) => s.predicted.length);
  const xLabels = solsWithIdx.map(({ i }) => `Top ${i + 1}`)
  const barColors = solsWithIdx.map(({ i }) => COLORS[i]);

  // CSS class `.alt-grid` reads --alt-cols; the media query collapses to a
  // single stacked column on narrow screens.
  const altCols = ALT_LBL.length ? ALT_LBL.map(() => "1fr").join(" ") : "1fr";

  // Track the first chart's width so all label text in every chart can scale
  // to fit one bar-column wide. All charts share the same column count and
  // (because the grid uses 1fr tracks) the same width, so one observer suffices.
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
  }, [ALT_LBL.length]);

  // Inner plot area ≈ chart width minus left+right margins (l:55 + r:15 = 70).
  // Per-column width = inner / nBars. Font sized so "100.0%" comfortably fits.
  const nBars = solsWithIdx.length || 1;
  const innerWidth = Math.max(0, chartWidth - 70);
  const colWidth = innerWidth / nBars;
  const fontSize = Math.max(9, Math.min(15, Math.round(colWidth * 0.22)));

  return (
    <div className="section">
      <h1>Predicted Vs. Observed Shares</h1>

      {data?.solutions.length === 1 && (
        <div class="card">
          <MiniSummaryTable s={data?.solutions[0]} />
        </div>
      )} {data?.solutions.length > 1 && (
          <SummaryTable sols={sols} />
      )}

      <div className="card" ref={cardRef}>
        <div className="section-toolbar no-export">
          <div className="btn-group">
            <button onClick={handleExport} disabled={!ALT_LBL.length || saving}>
              {saving ? "Saving…" : "Save as PNG"}
            </button>
          </div>
        </div>
        <div className="alt-grid" style={{ "--alt-cols": altCols }} ref={gridRef}>
          {ALT_LBL.map((altName, j) => {
            const predPct = solsWithIdx.map(({ s }) => +(s.predicted[j] * 100).toFixed(1));
            const obsValue = obs ? +(obs[j] * 100).toFixed(1) : null;
            const all = [...predPct, ...(obsValue !== null ? [obsValue] : [])];
            const yMin = all.length ? Math.floor(Math.min(...all) * 0.9) : 0;
            const yMax = predPct.length ? Math.ceil(Math.max(...predPct) * 1.1) : 70;
            const shapes = obsValue !== null ? [{
              type: "line", xref: "paper", yref: "y",
              x0: 0, x1: 1, y0: obsValue, y1: obsValue,
              line: { color: PLOT_THEME.obsLine, width: 2.5, dash: "dot" },
            }] : [];
            const annotations = obsValue !== null ? [{
              xref: "paper", x: 0.98, xanchor: "right",
              y: obsValue, yshift: 10,
              text: `Obs: ${obsValue}%`,
              showarrow: false, font: { size: fontSize, color: PLOT_THEME.obsLine },
            }] : [];
            xLabels.forEach((lbl, k) => {
              annotations.push({
                xref: "x", yref: "y",
                x: lbl, y: yMin, yshift: 4, yanchor: "bottom",
                text: `${predPct[k]}%`,
                showarrow: false, font: { size: fontSize, color: PLOT_THEME.titleColor },
              });
            });
            return (
              <Plot
                key={altName}
                style={{ height: 360 }}
                data={[{
                  x: xLabels, y: predPct, type: "bar",
                  marker: { color: barColors, opacity: 0.85 },
                  hovertemplate: `${altName} — %{x}: %{y:.1f}%<extra></extra>`,
                }]}
                layout={{
                  ...PLOT_THEME,
                  title: { text: altName, font: { size: fontSize, color: PLOT_THEME.titleColor } },
                  showlegend: false,
                  xaxis: { ...PLOT_THEME.xaxis, showticklabels: solsWithIdx.length > 1 },
                  yaxis: { ...PLOT_THEME.yaxis, title: "Share (%)", range: [yMin, yMax] },
                  margin: { t: 30, b: 50, l: 55, r: 15 },
                  shapes, annotations,
                }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
