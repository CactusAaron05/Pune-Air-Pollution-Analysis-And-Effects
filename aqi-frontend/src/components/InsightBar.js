function InsightBar({ data }) {

  const aqi = Math.round(data.predicted_aqi["1h"]);
  const risk = data.health_risk["1h"].interpretation.risk_level;
  const cause = data.causes["1h"].primary_source.source.replace("_", " ");

  return (
    <div className="insight-bar">

      <div>
        AQI is <strong>{aqi}</strong> and expected to worsen.
      </div>

      <div>
        Primary cause: <strong>{cause}</strong>
      </div>

      <div>
        Health risk: <strong>{risk}</strong>
      </div>

    </div>
  );
}

export default InsightBar;