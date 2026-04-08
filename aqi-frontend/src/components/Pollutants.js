function Pollutants({ data, health }) {

  const contribution = health?.["1h"]?.pollutant_contribution || {};
  const dominant = health?.["1h"]?.dominant_pollutant;

  return (
    <div className="pollutants">

      {Object.keys(data).map((key) => {

        const p = data[key];

        const percent = Math.round(
          contribution[key] ?? 
          contribution[key.replace(".", "")] ?? 
          0
        );

        return (
          <div
            className={`pollutant-card ${
              key === dominant ? "dominant" : ""
            } ${percent === 0 ? "inactive" : ""}`}
            key={key}
          >

            {/* HEADER */}
            <div className="pollutant-header">
              {key} {key === dominant && "⭐"}
            </div>

            {/* BAR */}
            <div className="pollutant-bar-container">
              <div
                className="pollutant-bar"
                style={{ width: `${percent}%` }}
              ></div>
            </div>

            {/* PERCENT */}
            <div className="pollutant-percent">
              {percent}%
            </div>

            {/* VALUES */}
            <div className="pollutant-values">

              <div className="pollutant-row">
                <span>+1h</span>
                <span>{Math.round(p["1h"])}</span>
              </div>

              <div className="pollutant-row">
                <span>+3h</span>
                <span>{Math.round(p["3h"])}</span>
              </div>

              <div className="pollutant-row">
                <span>+6h</span>
                <span>{Math.round(p["6h"])}</span>
              </div>

            </div>

          </div>
        );
      })}

    </div>
  );
}

export default Pollutants;