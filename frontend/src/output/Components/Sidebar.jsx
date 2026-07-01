/*
  Dark navigation sidebar. The model summary block and which nav items appear
  (Convergence / Distributions / Correlations) depend on flags in the payload,
  matching the conditional nav the original generator produced.
*/
export default function Sidebar({ data, section, setSection, drawerOpen, onClose }) {
  const items = [
    { key: "summary", label: "📊 Model Comparison", show: data?.solutions.length > 1 },
    { key: "coefficients", label: "📋 Coefficients", show: true },
    { key: "convergence", label: "📈 Convergence", show: data?.flags.hasConvergence },
    { key: "distributions", label: "🔔 Distributions", show: data?.flags.hasRandom },
    { key: "correlations", label: "🔗 Correlations", show: data?.flags.hasCorvars && data?.solutions.length > 1 },
    { key: "shares", label: "🎯 Predicted Shares", show: true },
  ];

  const top = data?.solutions?.[0];

  return (
    <div id="sidebar" className={drawerOpen ? "open" : ""}>
      <div className="sb-head">
        <div className="sb-model">
          {top ? (
            <>
              <b>{top.model}</b><br />
              Run: {data.runId} <br />
              {data.solutions.length > 1 && <>Obj: {data.objective}<br /></>}
              Alts: {top.n_alts}<br />
              Individuals: {data.individuals}<br />
              Choices per Individual: {data.choicesPerIndividual}<br />
              Total Choices: {data.totalChoices}<br />
              Draws: {data.draws}
            </>
          ) : "—"}
        </div>
      </div>
      {items.filter((i) => i.show).map((i) => (
        <div
          key={i.key}
          className={`nav-item ${section === i.key ? "active" : ""}`}
          onClick={() => { setSection(i.key); onClose && onClose(); }}
        >
          {i.label}
        </div>
      ))}
    </div>
  );
}
