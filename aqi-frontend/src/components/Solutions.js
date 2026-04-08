function formatWhy(why) {
  return why?.replaceAll("_", " ").replace("+", " + ") || "Unknown";
}

function Solutions({ data }) {

  if (!data) return null;

  return (
    <div className="section">
    <div className="solutions">

      <div className="solutions-context">
        <div>{data.where} | {data.when}</div>
        <div>⚠ {formatWhy(data.why)}</div>
      </div>

      <div className="solutions-block">
        <h4>Immediate Actions</h4>
        {data.short_term.map((s, i) => (
          <div key={i} className="solution-item primary">{s}</div>
        ))}
      </div>

      <div className="solutions-block">
        <h4>Long-term Strategy</h4>
        {data.long_term.map((s, i) => (
          <div key={i} className="solution-item secondary">{s}</div>
        ))}
      </div>

    </div>
    </div>
  );
}

export default Solutions;