import Plot from "./Plot.jsx";
import { COLORS, usePlotTheme } from "./helpers.js";

export default function MiniSummary({ s }) {
  /*
    table with the same data as Summary in Summary.jsx bot only one row
  */
  const PLOT_THEME = usePlotTheme();

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

  return (
    <>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>BIC</th><th>AIC</th><th>LogLik</th><th>Adj.ρ²</th>
              <th>#Fixed</th><th>#Random</th><th>#Corvars</th><th>Convergence</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>{Math.round(s.bic)}</b></td>
              <td>{s.aic ? Math.round(s.aic) : "-"}</td>
              <td>{s.loglik ? Math.round(s.loglik) : "-"}</td>
              <td>{s.adjlik ? s.adjlik.toFixed(3) : "-"}</td>
              <td>{s.fixed.length}</td>
              <td>{s.random.length}</td>
              <td>{s.corrvars.length}</td>
              <td>{convergenceCell(s)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </>
  );
}
