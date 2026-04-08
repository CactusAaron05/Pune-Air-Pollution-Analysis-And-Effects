function AQITrend({ data }) {
  return (
    <div className="aqi-trend">

      <div className="aqi-trend-card">
        <div className="aqi-trend-label">Next Hour</div>
        <div className="aqi-trend-value">{Math.round(data["1h"])}</div>
      </div>

      <div className="aqi-trend-card">
        <div className="aqi-trend-label">Short-term</div>
        <div className="aqi-trend-value">{Math.round(data["3h"])}</div>
      </div>

      <div className="aqi-trend-card">
        <div className="aqi-trend-label">Later Today</div>
        <div className="aqi-trend-value">{Math.round(data["6h"])}</div>
      </div>

    </div>
  );
}

export default AQITrend;