import { useState, Fragment } from "react";
import { sigCls } from "./helpers.js";
import MiniSummary from "./MiniSummary.jsx"

/*
  Coefficients: per-solution parameter table (intercepts / fixed / random with
  sd sub-rows) plus goodness-of-fit chips and, when present, an inline
  correlation matrix. Ported from renderCoeff.
*/
export default function Coefficients({ data }) {
  const sols = data.solutions;
  const [idx, setIdx] = useState(0);
  const sol = sols[idx];
  if (!sol) return null;

  const paramRow = (p) => (
    <tr key={p.var}>
      <td><b>{p.var}{sol.bcvars.includes(p.var) && <span title="BoxCox" className="var-link"> 📦</span>}</b></td>
      <td className="t-center">—</td>
      <td className="t-center">{p.coeff.toFixed(4)}</td>
      <td className="t-center">{p.se.toFixed(4)}</td>
      <td className="t-center">{p.zval.toFixed(3)}</td>
      <td className="t-center">{p.pval.toFixed(4)}</td>
      <td className={`t-center ${sigCls(p.sig)}`}>{p.sig}</td>
    </tr>
  );

  return (
    <div className="section">
      <h1>Coefficients</h1>
      <div className="card">
        { sols.length > 1 && (
          <div className="controls">
            <label>Solution:</label>
            <select value={idx} onChange={(e) => setIdx(parseInt(e.target.value))}>
              {sols.map((s, i) => (
                <option key={i} value={i}>Top {i + 1}  (BIC={Math.round(s.bic)})</option>
              ))}
            </select>
          </div>
        )}
      
        <MiniSummary s={sol} />
        <div style={{height: "25px"}} />

        <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Variable</th>
              <th className="t-center">Dist</th><th className="t-center">Coefficient</th><th className="t-center">SE</th>
              <th className="t-center">z-val</th><th className="t-center">p-val</th><th className="t-center">Sig</th>
            </tr>
          </thead>
          <tbody>
            {sol.intercepts && sol.intercepts.length > 0 && (
              <>
                <tr className="sec-row"><td colSpan={7}>Intercepts</td></tr>
                {sol.intercepts.map(paramRow)}
              </>
            )}
            {sol.fixed.length > 0 && (
              <>
                <tr className="sec-row"><td colSpan={7}>Fixed Parameters</td></tr>
                {sol.fixed.map(paramRow)}
              </>
            )}
            {sol.random.length > 0 && (
              <>
                <tr className="sec-row"><td colSpan={7}>Random Parameters</td></tr>
                {sol.random.map((p) => (
                  <Fragment key={p.var}>
                    <tr>
                      <td>
                        <b>{p.var}{sol.corrvars.includes(p.var) && <span title="Correlated" className="var-link"> 🔗</span>}{sol.bcvars.includes(p.var) && <span title="BoxCox" className="var-link"> 📦</span>}</b>
                      </td>
                      <td className="t-center">{p.dist} (mean)</td>
                      <td className="t-center">{p.mean.toFixed(4)}</td>
                      <td className="t-center">{p.se_mean.toFixed(4)}</td>
                      <td className="t-center">{p.zval.toFixed(3)}</td>
                      <td className="t-center">{p.pval.toFixed(4)}</td>
                      <td className={`t-center ${sigCls(p.sig)}`}>{p.sig}</td>
                    </tr>
                    {p.sd !== null && (
                      <tr className="coeff-sd-row">
                        <td>sd.{p.var}</td>
                        <td className="t-center">{p.dist} (sd)</td>
                        <td className="t-center">{p.sd.toFixed(4)}</td>
                        <td className="t-center">{p.se_sd !== null ? p.se_sd.toFixed(4) : "-"}</td>
                        <td className="t-center">{p.zval_sd !== null && p.zval_sd !== undefined ? p.zval_sd.toFixed(3) : "—"}</td>
                        <td className="t-center">{p.pval_sd !== null && p.pval_sd !== undefined ? p.pval_sd.toFixed(4) : "—"}</td>
                        <td className={`t-center ${sigCls(p.sig_sd || "-")}`}>{p.sig_sd || "-"}</td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </>
            )}
          </tbody>
        </table>
        </div>
        <div className="table-scroll">
          <table> {/* Legend is in a table to make it look nice*/}
            <thead>
              <th> 
                <td>Correlated Variables 🔗 · BoxCox Variables 📦</td>
              </th>
            </thead>
          </table>
        </div>

        {sol.corrvars.length > 0 && (() => {
          let maxPos = 0, minNeg = 0;
          sol.corrvars.forEach((v, row) => {
            sol.corrvars.forEach((_, col) => {
              if (col < row && sol.correlations[v] && sol.correlations[v][col]) {
                const val = parseFloat(sol.correlations[v][col][0]);
                if (Number.isNaN(val)) return;
                if (val > maxPos) maxPos = val;
                if (val < minNeg) minNeg = val;
              }
            });
          });
          return (
          <div className="table-scroll">
          <table className="matrix-table coeff-corr">
            <tbody>
              <tr className="sec-row"><td colSpan={sol.corrvars.length + 1}>Correlation Matrix</td></tr>
              <tr>
                <td />
                {sol.corrvars.map((v) => (
                  <td key={v} className="matrix-hdr">{v}</td>
                ))}
              </tr>
              {sol.corrvars.map((v, row) => (
                <tr key={v}>
                  <td className="matrix-hdr">{v}</td>
                  {sol.corrvars.map((_, col) => {
                    if (col === row) {
                      return <td key={col} className="matrix-diag">—</td>;
                    }
                    if (col < row && sol.correlations[v] && sol.correlations[v][col]) {
                      const [rawVal, sig] = sol.correlations[v][col];
                      const val = parseFloat(rawVal);
                      const posIntensity = maxPos !== 0 ? 2.7**(3*(val/maxPos-1)) * 0.6 : 0;
                      const negIntensity = minNeg !== 0 ? 2.7**(3*(val/minNeg-1)) * 0.7 : 0;
                      const bg = val > 0 ? `rgba(74,222,128,${posIntensity})` : `rgba(222,4,4,${negIntensity})`;
                      return (
                        <td key={col} className="matrix-cell" style={{ background: bg }}>
                          <b>{val.toFixed(3)}</b> <span className={sigCls(sig)}>({sig})</span>
                        </td>
                      );
                    }
                    return <td key={col} className="matrix-empty">—</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          </div>
          );
        })()}
      </div>
    </div>
  );
}
