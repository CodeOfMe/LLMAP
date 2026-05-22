import json

HARD_SKILL_SAFEGUARDS = {
    "confirmation_required": True,
    "audit_log": True,
    "rollback_plan": True,
    "rate_limit": "10/min",
    "max_impact": "single_record",
}

if __name__ == "__main__":
    print(json.dumps(HARD_SKILL_SAFEGUARDS, indent=2, ensure_ascii=False))