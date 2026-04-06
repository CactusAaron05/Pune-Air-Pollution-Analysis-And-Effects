import json
import os
from datetime import datetime

# Load pattern dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "source_patterns.json")

with open(DATA_PATH, "r") as f:
    SOURCE_PATTERNS = json.load(f)


def detect_causes(predictions):

    total = sum(predictions.values()) + 1e-6
    normalized = {k: v / total for k, v in predictions.items()}

    scores = []

    for source, config in SOURCE_PATTERNS.items():
        pollutants = config["pollutants"]

        score = sum(normalized.get(p, 0) for p in pollutants)

        scores.append({
            "source": source,
            "description": config["description"],
            "score": score
        })

    top_sources = sorted(scores, key=lambda x: x["score"], reverse=True)

    primary = top_sources[0]
    secondary = top_sources[1] if len(top_sources) > 1 else None

    return {
        "primary_source": {
            "source": primary["source"],
            "description": primary["description"],
            "confidence": round(primary["score"], 3)
        },
        "secondary_source": {
            "source": secondary["source"],
            "description": secondary["description"],
            "confidence": round(secondary["score"], 3)
        } if secondary else None
    }