function getAQIClass(aqi) {
  if (aqi <= 50) return "aqi-good";
  if (aqi <= 100) return "aqi-moderate";
  if (aqi <= 200) return "aqi-unhealthy";
  if (aqi <= 300) return "aqi-very-unhealthy";
  return "aqi-hazardous";
}

function AQICard({ aqi }) {
  const aqiClass = getAQIClass(aqi);

  return (
    <div className={`aqi-card ${aqiClass}`}>
      <div className="aqi-card-title">
        Predicted AQI (1h)
      </div>

      <div className="aqi-card-value">
        {Math.round(aqi)}
      </div>
    </div>
  );
}

export default AQICard;