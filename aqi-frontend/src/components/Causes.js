import React from "react";
import visualMap from "../utils/visualMapping";

function formatSource(source) {
  return source
    .replaceAll("_", " ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

function CauseBlock({ title, cause }) {

  if (!cause) return null;

  const primary = cause.primary_source;
  const secondary = cause.secondary_source;

  const primaryVisual = visualMap[primary.source] || {};
  const secondaryVisual = visualMap[secondary?.source] || {};

  return (
    <div className="cause-card">
      
      <div className="cause-header">{title}</div>

      <div className="cause-primary">
        <div className="cause-label">
          🔍 {primaryVisual.label || formatSource(primary.source)}
        </div>

        <div className="cause-confidence">
          {Math.round(primary.confidence * 100)}%
        </div>
      </div>

      {secondary && (
        <div className="cause-secondary">
          {secondaryVisual.label || formatSource(secondary.source)} (
          {Math.round(secondary.confidence * 100)}%)
        </div>
      )}

    </div>
  );
}

function Causes({ data }) {

  if (!data) return null;

  return (
    <div className="causes">

      <CauseBlock title="Now (1h)" cause={data["1h"]} />
      <CauseBlock title="Soon (3h)" cause={data["3h"]} />
      <CauseBlock title="Later (6h)" cause={data["6h"]} />

    </div>
  );
}

export default Causes;