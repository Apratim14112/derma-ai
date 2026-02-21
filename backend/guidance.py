import re
from typing import Any, Dict, List

DISCLAIMER = (
    "This AI tool is for educational use only and is not medical advice. "
    "Skin conditions can look similar. If you are concerned or symptoms worsen, "
    "seek care from a licensed clinician."
)

GENERAL_WELLNESS = [
    "Use broad-spectrum SPF 30+ daily and reapply when outdoors.",
    "Avoid tanning beds and minimize midday sun exposure.",
    "Keep skin moisturized and avoid harsh, fragranced products.",
    "Perform monthly skin self-checks and note changes in size, color, or shape.",
    "Stay hydrated and support skin health with balanced nutrition.",
]

OTC_SUGGESTIONS_GENTLE = [
    "Broad-spectrum sunscreen (SPF 30+).",
    "Gentle, fragrance-free moisturizer.",
    "Mild cleanser; avoid abrasive scrubs.",
    "Petrolatum (petroleum jelly) for barrier support.",
]

OTC_SUGGESTIONS_ITCH = [
    "Cool compress for itching or irritation.",
    "OTC 1% hydrocortisone for short-term itch relief (avoid open wounds).",
    "Non-drowsy oral antihistamine for itch (if appropriate for you).",
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _risk_bucket(label: str) -> str:
    text = _normalize(label)
    if any(term in text for term in ["melanoma", "carcinoma", "cancer"]):
        return "high"
    if any(term in text for term in ["actinic", "keratosis", "keratoses", "dysplasia"]):
        return "moderate"
    if any(term in text for term in ["nevus", "nevi", "benign", "vascular", "dermatofibroma"]):
        return "low"
    return "unknown"


def build_guidance(label: str, confidence: float) -> Dict[str, Any]:
    risk = _risk_bucket(label)
    warnings: List[str] = [DISCLAIMER]

    if confidence < 0.55:
        warnings.append(
            "The model confidence is low. Consider a clearer image or consult a clinician."
        )

    if risk == "high":
        otc = []
        next_steps = [
            "Do not self-treat. Schedule an in-person evaluation with a dermatologist.",
            "Seek urgent care if the lesion is rapidly changing, bleeding, or painful.",
        ]
    elif risk == "moderate":
        otc = OTC_SUGGESTIONS_GENTLE
        next_steps = [
            "Monitor for changes and consider a clinician visit, especially if the lesion is new.",
            "Use sun protection consistently to reduce further damage.",
        ]
    else:
        otc = OTC_SUGGESTIONS_GENTLE + OTC_SUGGESTIONS_ITCH
        next_steps = [
            "If symptoms persist beyond 2-3 weeks or worsen, consult a clinician.",
            "Avoid picking or scratching the area.",
        ]

    return {
        "label": label,
        "risk": risk,
        "otc_suggestions": otc,
        "next_steps": next_steps,
        "wellness_tips": GENERAL_WELLNESS,
        "warnings": warnings,
    }
