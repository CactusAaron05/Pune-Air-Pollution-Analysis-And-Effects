import visualMap from "../utils/visualMapping";

function Causes({ data }) {

  // 🔥 Use ONLY 1h (current state)
  const cause = data?.["1h"];

  if (!cause) return null;

  const primary = cause.primary_source;
  const secondary = cause.secondary_source;

  const primaryVisual = visualMap[primary.source] || {};
  const secondaryVisual = visualMap[secondary.source] || {};

  return (
    <div className="causes">

      {/* 🔴 PRIMARY */}
      <div className="cause-card primary">

        <div className="cause-header">Primary Cause</div>

        <div className="cause-main">

          <div className="cause-icon">
            {primaryVisual.icon || "⚠️"}
          </div>

          <div className="cause-info">

            <div className="cause-label">
              {primaryVisual.label || primary.source}
            </div>

            {/* CONFIDENCE BAR */}
            <div className="cause-confidence-bar">
              <div
                className="cause-confidence-fill"
                style={{ width: `${primary.confidence * 100}%` }}
              ></div>
            </div>

            <div className="cause-confidence-text">
              {(primary.confidence * 100).toFixed(1)}% confidence
            </div>

          </div>

        </div>

      </div>

      {/* 🟠 SECONDARY */}
      <div className="cause-card secondary">

        <div className="cause-header">Secondary Cause</div>

        <div className="cause-main">

          <div className="cause-icon">
            {secondaryVisual.icon || "ℹ️"}
          </div>

          <div className="cause-info">

            <div className="cause-label">
              {secondaryVisual.label || secondary.source}
            </div>

            <div className="cause-confidence-bar">
              <div
                className="cause-confidence-fill secondary"
                style={{ width: `${secondary.confidence * 100}%` }}
              ></div>
            </div>

            <div className="cause-confidence-text">
              {(secondary.confidence * 100).toFixed(1)}% confidence
            </div>

          </div>

        </div>

      </div>

    </div>
  );
}

export default Causes;