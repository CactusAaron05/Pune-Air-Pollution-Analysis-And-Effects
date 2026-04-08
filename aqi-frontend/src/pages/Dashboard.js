import { useEffect, useState } from "react";
import { getDashboard } from "../services/api";

import AQIHero from "../components/AQIHero";
import AQIGraph from "../components/AQIGraph";
import Pollutants from "../components/Pollutants";
import Causes from "../components/Causes";
import Health from "../components/Health";
import Solutions from "../components/Solutions";
import Alerts from "../components/Alerts";
import AccordionSection from "../components/AccordionSection";
import InsightBar from "../components/InsightBar";



function Dashboard() {

  const [region, setRegion] = useState("Hadapsar");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const regions = [
    "Hadapsar",
    "Bhosari",
    "Nigdi",
    "Katraj",
    "Karve Road",
    "Pashan",
    "Alandi"
  ];

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const res = await getDashboard(region);
        setData(res);
      } catch (err) {
        console.error(err);
        setData(null);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [region]);

  if (loading) return <div className="loading">Loading...</div>;
  if (!data) return <div className="error">Error loading data</div>;

  return (
    <div className="dashboard">

      {/* 🔴 REGION SELECTOR (RESTORED) */}
      <div className="region-selector">
        <label>Select Region:</label>
        <select
          value={region}
          onChange={(e) => setRegion(e.target.value)}
        >
          {regions.map((r) => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
      </div>

      {/* HERO */}
      <AQIHero data={data.predicted_aqi} />

      {/* GRAPH */}
      <AQIGraph data={data.predicted_aqi} />

      {/* ALERTS */}
      <Alerts data={data.alerts} />

      {/* SECTIONS */}
      <AccordionSection title="Pollutants" defaultOpen>
        <Pollutants
          data={data.predicted_pollutants}
          health={data.health_risk}
        />
      </AccordionSection>

      <AccordionSection title="Causes">
        <Causes data={data.causes} />
      </AccordionSection>

      <AccordionSection title="Health">
        <Health data={data.health_risk} />
      </AccordionSection>

      <AccordionSection title="Solutions">
        <Solutions data={data.solutions} />
      </AccordionSection>

      <InsightBar data={data} />

    </div>
  );
}

export default Dashboard;