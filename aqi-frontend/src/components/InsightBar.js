import React from "react";

function InsightBar({ data }) {

  if (!data) return null;

  const aqi = data.predicted_aqi;
  const health = data.health_risk;
  const causes = data.causes;

  if (!aqi || !health || !causes) return null;

  // ── STEP 1: TREND DETECTION ──
  const a1 = aqi["1h"];
  const a3 = aqi["3h"];
  const a6 = aqi["6h"];

  let trendText = "stable";

  if (a1 < a3 && a3 < a6) {
    trendText = "worsening";
  } else if (a1 > a3 && a3 > a6) {
    trendText = "improving";
  }

  // ── STEP 2: HEALTH CONTEXT ──
  const h1 = health["1h"];
  const riskLevel = h1?.interpretation?.risk_level || "Unknown";
  const riskIncrease = Math.round(h1?.interpretation?.risk_increase || 0);

  // ── STEP 3: CAUSE CONTEXT ──
  const cause1h = causes["1h"];
  const primarySource = cause1h?.primary_source?.source || "unknown";

  const formattedSource = primarySource
    .replaceAll("_", " ")
    .replace(/\b\w/g, c => c.toUpperCase());

  // ── STEP 4: BUILD MESSAGE ──
  let message = "";

  if (trendText === "worsening") {
    message = `AQI is ${Math.round(a1)} and worsening. Primary cause: ${formattedSource}. Health risk is ${riskLevel} (+${riskIncrease}%).`;
  } 
  else if (trendText === "improving") {
    message = `AQI is ${Math.round(a1)} but improving. Primary cause: ${formattedSource}. Health risk remains ${riskLevel} (+${riskIncrease}%).`;
  } 
  else {
    message = `AQI is ${Math.round(a1)} and stable. Primary cause: ${formattedSource}. Health risk is ${riskLevel} (+${riskIncrease}%).`;
  }

  return (
    <div className="insight-bar">
      {message}
    </div>
  );
}

export default InsightBar;