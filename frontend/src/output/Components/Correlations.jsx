import { sigCls } from "./helpers.js";

/*
  Correlation Matrix: lower-triangular table over the union of correlated
  variables. Each cell stacks one mini-row per Top solution (T1..Tn) showing
  that solution's correlation estimate and significance. Ported from renderCorr.
*/

// Look up correlation (value, sig) between v1 and v2 in one solution.
function lookup(s, v1, v2) {
  const vars = s.corrvars || [];
  const ri = vars.indexOf(v1), ci = vars.indexOf(v2);
  if (ri < 0 || ci < 0) return [null, null];
  const lo = Math.max(ri, ci), hi = Math.min(ri, ci);
  const corrs = s.correlations[vars[lo]];
  if (corrs && corrs[hi]) return [parseFloat(corrs[hi][0]), corrs[hi][1]];
  return [null, null];
}

export default function Correlations({ data }) {
  const sols = data.solutions;

  const varSet = new Set(), varList = [];
  sols.forEach((s) => (s.corrvars || []).forEach((v) => {
    if (!varSet.has(v)) { varSet.add(v); varList.push(v); }
  }));

  // use the max positive and min negitive to give the correlation table a max intensity for colors
  let maxPos = 0, minNeg = 0;
  sols.forEach((s) => {
    const vars = s.corrvars || [];
    for (let i = 0; i < vars.length; i++) {
      for (let j = 0; j < i; j++) {
        const [val] = lookup(s, vars[i], vars[j]);
        if (val === null || Number.isNaN(val)) continue;
        if (val > maxPos) maxPos = val;
        if (val < minNeg) minNeg = val;
      }
    }
  });

  if (!varList.length) {
    return (
      <div className="section">
        <h1>Correlation Matrix</h1>
        <div className="card"><p className="empty">No correlation structure.</p></div>
      </div>
    );
  }

  const cellStack = (v1, v2) => varList && sols.map((s, i) => {
    const [val, sig] = lookup(s, v1, v2);
    if (val === null) {
      return (
        <span key={i} className="corr-stack corr-stack-empty">
          <span>T{i + 1}:</span><span>—</span><span />
        </span>
      );
    }
    // exponential funciton on the color intensity makes the values stand out more, especially with clustered values
    const posIntensity = maxPos !== 0 ? 2.7**(3*(val/maxPos-1)) * 0.6 : 0
    const negIntensity = minNeg !== 0 ? 2.7**(3*(val/minNeg-1)) * 0.7 : 0

    const bg = val > 0 ? `rgba(74,222,128,${posIntensity})` : `rgba(222,4,4,${negIntensity})`;
    return (
      <span key={i} className="corr-stack" style={{ background: bg }}>
        <span>T{i + 1}:</span>
        <span><b>{val.toFixed(3)}</b></span>
        <span className={sigCls(sig)}>{sig}</span>
      </span>
    );
  });

  return (
    <div className="section">
      <h1>Correlation Matrix</h1>
      <div className="card">
        <div className="table-scroll">
        <table className="corr-matrix matrix-table">
          <tbody>
            <tr>
              <td />
              {varList.map((v) => <td key={v} className="matrix-hdr">{v}</td>)}
            </tr>
            {varList.map((v1, rowIdx) => (
              <tr key={v1}>
                <td className="matrix-hdr">{v1}</td>
                {varList.map((v2, colIdx) => (
                  colIdx >= rowIdx ? (
                    <td key={v2} className="matrix-blank-cell">—</td>
                  ) : (
                    <td key={v2} className="matrix-stack-cell">
                      {cellStack(v1, v2)}
                    </td>
                  )
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      </div>
    </div>
  );
}
