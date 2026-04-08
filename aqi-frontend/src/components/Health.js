import { HeartPulse } from "lucide-react";


function Health({ data }) {

  const h1 = data?.["1h"];
  const h3 = data?.["3h"];
  const h6 = data?.["6h"];

  if (!h1) return null;

  const riskLevel = h1.interpretation?.risk_level;
  const riskIncrease = h1.interpretation?.risk_increase;
  const message = h1.interpretation?.message;

  const groups = h1.health_impact?.sensitive_groups || [];

  return (
    <div className="section">
    <div className="health">
    {/* <HeartPulse size={18} /> */}
      {/* 🔴 MAIN RISK CARD */}
      <div className="health-main">

        <div className="health-risk-level">
          {riskLevel} Risk
        </div>

        <div className="health-risk-increase">
          +{riskIncrease.toFixed(1)}%
        </div>

        <div className="health-message">
          {message}
        </div>

      </div>

      {/* 🧠 TREND (SMART INSIGHT) */}
      <div className="health-trend">

        <div className="health-trend-item">
          +3h: {h3?.interpretation?.risk_level}
        </div>

        <div className="health-trend-item">
          +6h: {h6?.interpretation?.risk_level}
        </div>

      </div>

      {/* 👥 SENSITIVE GROUPS */}
      <div className="health-section">

        <div className="health-section-title">
          Sensitive Groups
        </div>

        <div className="health-tags">
          {groups.map((g, i) => (
            <span key={i} className="health-tag">
              {g.replaceAll("_", " ")}
            </span>
          ))}
        </div>

      </div>

    </div>
    </div>
  );
}

export default Health;