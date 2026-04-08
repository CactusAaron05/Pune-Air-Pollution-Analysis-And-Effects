import { useState } from "react";

function AccordionSection({ title, summary, children, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="accordion">

      <div className="accordion-header" onClick={() => setOpen(!open)}>
        <strong>{title}</strong>
        <span>{summary}</span>
      </div>

      {open && <div className={`accordion-content ${open ? "open" : ""}`}>
  {children}
</div>}

    </div>
  );
}

export default AccordionSection;