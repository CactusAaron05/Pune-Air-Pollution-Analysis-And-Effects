import React from "react";

function Health({ data }) {

  if (!data) return null;

  const h1 = data["1h"];
  const h3 = data["3h"];
  const h6 = data["6h"];

  if (!h1 || !h3 || !h6) return null;

  // ── MAIN DATA ──
  const riskLevel = h1.interpretation.risk_level;
  const riskIncrease = Math.round(h1.interpretation.risk_increase);

  const dominant = h1.dominant_pollutant;
  const contribution = h1.pollutant_contribution || {};

  const shortTerm = h1.health_impact?.short_term_effects || [];
  const longTerm = h1.health_impact?.long_term_effects || [];
  const sensitive = h1.health_impact?.sensitive_groups || [];

  // ── TREND DATA ──
  const r1 = h1.interpretation.risk_increase;
  const r3 = h3.interpretation.risk_increase;
  const r6 = h6.interpretation.risk_increase;

  const values = [
    { label: "1h", value: r1 },
    { label: "3h", value: r3 },
    { label: "6h", value: r6 }
  ];

  const max = Math.max(r1, r3, r6);

  return (
    <div className="health">

      {/* MAIN */}
      <div className="health-main">

        <div className="health-risk-level">
          🫀 {riskLevel} Risk
        </div>

        <div className="health-risk-increase">
          +{riskIncrease}%
        </div>

        <div className="health-message">
          Dominant pollutant: <strong>{dominant}</strong>
        </div>

      </div>

      {/* TREND */}
      <div className="health-trend">
        {values.map((item, i) => {

          let type = "";

          if (item.value === max) type = "peak";
          else if (i === 0) type = "current";
          else type = "low";

          return (
            <div key={i} className={`health-trend-item ${type}`}>
  <span className="trend-time">{item.label}</span>
  <span className="trend-percent">{Math.round(item.value)}%</span>
</div>
          );
        })}
      </div>

      {/* CONTRIBUTION */}
      <div className="health-section">
        <div className="health-section-title">
          Pollutant Contribution
        </div>

        <div className="health-tags">
          {Object.entries(contribution).map(([k, v]) => {

            const isDominant = k === dominant;

            return (
              <div
                key={k}
                className={`health-tag ${isDominant ? "dominant" : ""}`}
              >
                {k}: {Math.round(v)}%
              </div>
            );
          })}
        </div>
      </div>

      {/* SHORT TERM */}
      <div className="health-section">
        <div className="health-section-title">
          Short-Term Effects
        </div>

        <div className="health-tags">
          {shortTerm.map((e, i) => (
            <div key={i} className="health-tag">
              {e}
            </div>
          ))}
        </div>
      </div>

      {/* LONG TERM */}
      <div className="health-section">
        <div className="health-section-title">
          Long-Term Effects
        </div>

        <div className="health-tags">
          {longTerm.map((e, i) => (
            <div key={i} className="health-tag">
              {e}
            </div>
          ))}
        </div>
      </div>

      {/* SENSITIVE */}
      <div className="health-section">
        <div className="health-section-title">
          Sensitive Groups
        </div>

        <div className="health-tags">
          {sensitive.map((g, i) => (
            <div key={i} className="health-tag">
              {g.replaceAll("_", " ")}
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}

export default Health;