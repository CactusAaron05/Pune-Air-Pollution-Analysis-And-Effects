import React from "react";

function classifyAlert(text) {
  const t = text.toLowerCase();

  if (t.includes("worsen") || t.includes("severe") || t.includes("hazard")) {
    return "critical";
  }

  if (t.includes("improv")) {
    return "neutral";
  }

  return "neutral";
}

function Alerts({ data }) {

  if (!data || data.length === 0) return null;

  return (
    <div className="alerts">

      <div className="alerts-header">🚨 Alerts</div>

      <div className="alerts-list">

        {data.map((alert, i) => {

          const type = classifyAlert(alert);

          return (
            <div key={i} className={`alert-item ${type}`}>
              <span style={{ marginRight: "6px" }}>
                {type === "critical" ? "🚨" : "⚠️"}
              </span>
              {alert}
            </div>
          );
        })}

      </div>

    </div>
  );
}

export default Alerts;