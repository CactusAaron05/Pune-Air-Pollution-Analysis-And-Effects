import { getAQIColor } from "../utils/aqiUtils";

function getTrend(aqi) {
  const { ["1h"]: h1, ["3h"]: h3, ["6h"]: h6 } = aqi;

  if (h6 > h3 && h3 > h1) return "↑ Rising";
  if (h6 < h3 && h3 < h1) return "↓ Improving";
  return "→ Stable";
}

function AQIHero({ data }) {
  const current = Math.round(data["1h"]);
  const color = getAQIColor(current);
  const trend = getTrend(data);

  return (
    <div className="aqi-hero" style={{ borderLeft: `6px solid ${color}` }}>

      <div className="aqi-hero-main">
        <div className="aqi-hero-value">{current}</div>

        <div>
          <div className="aqi-hero-label">{trend}</div>
        </div>
      </div>

      <div className="aqi-hero-timeline">
        <div>+1h: {Math.round(data["1h"])}</div>
        <div>+3h: {Math.round(data["3h"])}</div>
        <div>+6h: {Math.round(data["6h"])}</div>
      </div>

    </div>
  );
}

export default AQIHero;