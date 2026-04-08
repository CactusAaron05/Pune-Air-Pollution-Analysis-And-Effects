import { useState } from "react";
import { AlertTriangle } from "lucide-react";
function Alerts({ data }) {
  const [showRaw, setShowRaw] = useState(false);

  // 🔥 Normalize data safely
  let alertsArray = [];

  if (Array.isArray(data)) {
    alertsArray = data;
  } else if (typeof data === "string") {
    alertsArray = [data];
  } else if (data) {
    alertsArray = [JSON.stringify(data)];
  }

  return (
    <div className="alerts">

      <div className="alerts-header">
        ⚠️ Alerts
      </div>

      <div className="alerts-list">

        {alertsArray.length === 0 ? (
          <div className="alert-item neutral">
            No active alerts
          </div>
        ) : (
          alertsArray.map((alert, index) => (
            <div key={index} className="alert-item critical">
              {alert}
            </div>
          ))
        )}

      </div>

      <div className="alerts-toggle">
        <button onClick={() => setShowRaw(!showRaw)}>
          {showRaw ? "Hide Full Data" : "Show Full Data"}
        </button>
      </div>

      {showRaw && (
        <div className="alerts-raw">
          {alertsArray.map((alert, i) => (
            <div key={i} className="alerts-raw-item">
              <span>[{i}]</span>
              <span>{alert}</span>
            </div>
          ))}
        </div>
      )}

    </div>
  );
}

export default Alerts;