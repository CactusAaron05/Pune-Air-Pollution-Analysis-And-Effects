import React from "react";

function formatWhy(text) {
  return text
    .split("+")
    .map(s => s.trim())
    .map(s => s.replaceAll("_", " "))
    .map(s => s.replace(/\b\w/g, c => c.toUpperCase()))
    .join(" + ");
}

function Solutions({ data }) {

  if (!data) return null;

  const shortTerm = data.short_term || [];
  const longTerm = data.long_term || [];
  const explanations = data.explanation || [];

  return (
    <div className="solutions">

      <div className="solutions-context">

        <div>
          {data.where} | {data.when}
        </div>

        <div>
          Cause: {formatWhy(data.why)}
        </div>

      </div>

      <div className="solutions-block">

        <div>Short-Term Actions</div>

        {shortTerm.map((action, i) => {

          const explanation = explanations[i] || "";

          return (
            <div key={i} className={`solution-item ${i < 2 ? "primary" : "secondary"}`}>

              <div>🛠️ {action}</div>

              {explanation && (
                <div className="text-muted">
                  {explanation}
                </div>
              )}

            </div>
          );
        })}

      </div>

      <div className="solutions-block">

  <div>Long-Term Actions</div>

  {longTerm.map((action, i) => (
    <div key={i} className="solution-item secondary">
      <div>🌱 {action}</div>
    </div>
  ))}

</div>

    </div>
  );
}

export default Solutions;