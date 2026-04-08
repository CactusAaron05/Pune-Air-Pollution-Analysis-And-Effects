import React from "react";

function AQIHero({ data }) {

  // ✅ handle both possible shapes
  const aqi =
    data?.predicted_aqi ||
    data ||
    {};

  const a1 = aqi["1h"];
  const a3 = aqi["3h"];
  const a6 = aqi["6h"];

  const hasData =
    a1 !== undefined &&
    a3 !== undefined &&
    a6 !== undefined;

  const displayAQI = a1 !== undefined ? Math.round(a1) : "--";

  let trend = "Loading";
  let trendClass = "stable";

  if (hasData) {
    if (a1 < a3 && a3 < a6) {
      trend = "Worsening";
      trendClass = "worsening";
    } else if (a1 > a3 && a3 > a6) {
      trend = "Improving";
      trendClass = "improving";
    } else {
      trend = "Stable";
    }
  }

  return (
    <div className="aqi-hero">

      <div className="aqi-hero-value">
        {displayAQI}
      </div>

      <div className={`aqi-hero-trend ${trendClass}`}>
        {hasData
          ? (trend === "Improving" ? "↓" :
             trend === "Worsening" ? "↑" : "→") + " " + trend
          : "Loading..."}
      </div>

      {hasData && (
        <div className="aqi-mini-cards">

          <div className="aqi-mini-card current">
            <span>1h</span>
            <strong>{Math.round(a1)}</strong>
          </div>

          <div className="aqi-mini-card">
            <span>3h</span>
            <strong>{Math.round(a3)}</strong>
          </div>

          <div className="aqi-mini-card">
            <span>6h</span>
            <strong>{Math.round(a6)}</strong>
          </div>

        </div>
      )}

    </div>
  );
}

export default AQIHero;