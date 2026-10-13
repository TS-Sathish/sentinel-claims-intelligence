import os


def _minimized_features(features):
    # Only generic vehicle/incident descriptors are eligible for an optional
    # provider call. Unknown fields and all identifier-like fields are omitted.
    allowed = {"vehiclecategory", "vehicleprice", "ageofvehicle", "accidentarea",
               "days_policy_accident", "days_policy_claim", "pastnumberofclaims",
               "numberofsupplements", "numberofcars", "basepolicy", "polic ytype".replace(" ", "")}
    return {key: value for key, value in features.items()
            if key.lower().replace("_", "").replace(" ", "") in allowed
            and isinstance(value, (str, int, float, bool, type(None)))}


def narrative(result, features):
    if os.getenv("LLM_PROVIDER", "local").lower() == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=15, max_retries=0)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=.2,
                messages=[{"role": "system", "content": (
                    "Write a short, cautious claim review note with sections Summary, Model signals, "
                    "Limitations, and Verification steps. Scores are not proof. Never infer protected "
                    "traits, invent evidence, change scores, or recommend denial. State that a human "
                    "must verify evidence and make any decision.")},
                    {"role": "user", "content": f"Minimized claim descriptors: {_minimized_features(features)}\n"
                     f"Model signals: fraud_probability={result.get('fraud_probability')}; "
                     f"anomaly_flag={result.get('anomaly_flag')}; anomaly_score={result.get('anomaly_score')}. "
                     "Summarize these signals only; do not add facts."}],
                max_tokens=250)
            content = response.choices[0].message.content
            if content:
                return content[:3000]
        except Exception:
            pass
    probability = result.get("fraud_probability", 0)
    anomaly = "The anomaly model flagged an unusual pattern." if result.get("anomaly_flag") else "The anomaly model did not flag an unusual pattern."
    return (f"Summary: Preliminary review priority from the classifier is {probability:.1%}.\n"
            f"Model signals: {anomaly}\n"
            "Limitations: Model signals are not proof of fraud and may be wrong.\n"
            "Verification steps: Review submitted documents, policy details, and incident chronology. "
            "A qualified human reviewer must verify evidence and make any decision.")
