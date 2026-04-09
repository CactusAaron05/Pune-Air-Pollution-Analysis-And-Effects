import React from "react";

function Pollutants({ data, health }) {

  if (!data || !health) return null;

  const h1 = health["1h"];

  const contribution = h1?.pollutant_contribution || {};
  const dominant = h1?.dominant_pollutant;

  const pollutants = Object.keys(data);

  return (
    <div className="pollutants">

      {pollutants.map((key) => {

        const values = data[key] || {};

        const v1 = values["1h"] ?? 0;
        const v3 = values["3h"] ?? 0;
        const v6 = values["6h"] ?? 0;

        const percent = contribution[key] ?? 0;

        const isDominant = key === dominant;

        return (
          <div key={key} className={`pollutant-card ${isDominant ? "dominant" : ""}`}>

            <div className="pollutant-header">
              🌫️ {key}
            </div>

            <div className="pollutant-bar-container">
              <div className="pollutant-bar" style={{ width: `${Math.round(percent)}%` }} />
            </div>

            <div className="pollutant-percent">
              {Math.round(percent)}%
            </div>

            <div className="pollutant-row">1h: {Math.round(v1)}</div>
            <div className="pollutant-row">3h: {Math.round(v3)}</div>
            <div className="pollutant-row">6h: {Math.round(v6)}</div>

          </div>
        );
      })}

    </div>
  );
}

export default Pollutants;