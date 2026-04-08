export function getAQILevel(aqi) {
  if (aqi <= 50) return "good";
  if (aqi <= 100) return "moderate";
  if (aqi <= 200) return "poor";
  if (aqi <= 300) return "very-poor";
  return "severe";
}

export function getAQIColor(aqi) {
  if (aqi <= 50) return "#22c55e";       // green
  if (aqi <= 100) return "#eab308";      // yellow
  if (aqi <= 200) return "#f97316";      // orange
  if (aqi <= 300) return "#ef4444";      // red
  return "#a855f7";                      // purple
}