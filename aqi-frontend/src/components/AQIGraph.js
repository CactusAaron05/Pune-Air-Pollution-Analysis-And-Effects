import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";
import { getAQIColor } from "../utils/aqiUtils";

function AQIGraph({ data }) {

  const chartData = [
    { time: "+1h", value: Math.round(data["1h"]) },
    { time: "+3h", value: Math.round(data["3h"]) },
    { time: "+6h", value: Math.round(data["6h"]) }
  ];

  const color = getAQIColor(data["1h"]);

  return (
    <div className="aqi-graph">

      <div className="aqi-graph-title">AQI Trend</div>

      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData}>
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={3} />
        </LineChart>
      </ResponsiveContainer>

    </div>
  );
}

export default AQIGraph;