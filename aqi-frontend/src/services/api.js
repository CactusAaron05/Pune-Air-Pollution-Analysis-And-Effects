const BASE_URL = "http://127.0.0.1:8000";

// 🔹 Aggregated (initial load)
export const getDashboard = async (region) => {
  const res = await fetch(`${BASE_URL}/dashboard?region=${region}`);
  if (!res.ok) throw new Error("Dashboard API failed");
  return res.json();
};

// 🔹 Split endpoints (future use)
export const getForecast = async (region) => {
  const res = await fetch(`${BASE_URL}/aqi/forecast?region=${region}`);
  return res.json();
};

export const getPollutants = async (region) => {
  const res = await fetch(`${BASE_URL}/pollutants?region=${region}`);
  return res.json();
};

export const getCauses = async (region) => {
  const res = await fetch(`${BASE_URL}/causes?region=${region}`);
  return res.json();
};

export const getHealth = async (region) => {
  const res = await fetch(`${BASE_URL}/health?region=${region}`);
  return res.json();
};

export const getSolutions = async (region) => {
  const res = await fetch(`${BASE_URL}/solutions?region=${region}`);
  return res.json();
};

export const getAlerts = async (region) => {
  const res = await fetch(`${BASE_URL}/alerts?region=${region}`);
  return res.json();
};