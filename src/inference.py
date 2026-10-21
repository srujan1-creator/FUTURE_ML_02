"""
src/inference.py
================
Production ticket triage inference function with deterministic SLA guardrails.
"""

import re
import numpy as np
from src.preprocessing import preprocess_ticket_text

# Critical emergency patterns that trigger instant SLA high-priority escalation
CRITICAL_SLA_PATTERNS = [
    r'\bproduction (?:down|offline|halted|unavailable)\b',
    r'\b(?:security|data) breach\b',
    r'\bunauthorized (?:charge|access|transaction)\b',
    r'\b(?:500|502|503|504) internal server error\b',
    r'\bhipaa audit\b',
    r'\bgdpr (?:penalty|breach|deletion|dpa)\b',
    r'\b2fa (?:lost|lockout|fail)\b',
    r'\bdatabase connection timeout\b'
]
COMPILED_GUARDRAILS = [re.compile(p, re.IGNORECASE) for p in CRITICAL_SLA_PATTERNS]

QUEUE_ROUTING_MAP = {
    "Billing": "Tier-1 / Billing & Accounts Receivable",
    "Technical Issue": "Tier-2 / Engineering Operations",
    "Account": "Tier-1 / Identity & Security Support",
    "General Query": "Tier-1 / Product Specialists"
}


def predict_ticket(
    subject: str,
    body: str,
    cat_pipeline,
    pri_pipeline,
    confidence_threshold: float = 0.50
) -> dict:
    """
    Classifies category, predicts priority, and evaluates SLA emergency rules.
    """
    raw_combined = f"{subject or ''} {body or ''}".strip()
    if not raw_combined:
        return {"error": "Empty ticket contents", "routing_action": "REJECT_OR_PROMPT_USER"}
    
    cleaned = preprocess_ticket_text(raw_combined)
    
    # Category Inference
    cat_pred = cat_pipeline.predict([cleaned])[0]
    cat_probas = cat_pipeline.predict_proba([cleaned])[0]
    cat_conf = float(np.max(cat_probas))
    
    # Priority Inference
    pri_pred = pri_pipeline.predict([cleaned])[0]
    pri_probas = pri_pipeline.predict_proba([cleaned])[0]
    pri_conf = float(np.max(pri_probas))
    
    # Deterministic SLA Check
    sla_triggered = False
    matched_rule = None
    for pattern in COMPILED_GUARDRAILS:
        match = pattern.search(raw_combined)
        if match:
            sla_triggered = True
            matched_rule = match.group(0)
            break
            
    final_priority = "High" if sla_triggered else pri_pred
    
    requires_human_triage = (
        (cat_conf < confidence_threshold) or 
        (pri_conf < confidence_threshold) or 
        sla_triggered
    )
    
    return {
        "ticket_subject": subject,
        "predicted_category": cat_pred,
        "category_confidence": round(cat_conf, 4),
        "predicted_priority": final_priority,
        "priority_confidence": round(pri_conf, 4),
        "recommended_queue": QUEUE_ROUTING_MAP.get(cat_pred, "General Support"),
        "sla_guardrail_override": sla_triggered,
        "matched_guardrail_rule": matched_rule,
        "human_triage_recommended": requires_human_triage,
        "cleaned_preview": cleaned[:80] + ("..." if len(cleaned) > 80 else "")
    }
