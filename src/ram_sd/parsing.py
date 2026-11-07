"""Model-output errors are failures, never negative-class predictions."""

import json
import math
import re

PLANS = ("expectation_violation", "knowledge_dependent", "simple_irony")


def json_object(text):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON: {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    content = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if fence:
        content = fence.group(1).strip()
    content = re.sub(r"^(?:<<LABEL>>|«LABEL»)[^\n]*\n", "", content, count=1).strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if fence:
        content = fence.group(1).strip()
    value = json.loads(content, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError("Expected one complete JSON object in model output")
    return value


def probability(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field} must be a finite number in [0,1]")
    return float(value)


def nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def planner_output(text, require_entities=False):
    value = json_object(text)
    if value.get("selected_plan") not in PLANS:
        raise ValueError("Unknown or missing selected_plan")
    probability(value.get("confidence"), "confidence")
    nonempty(value.get("contextual_analysis"), "contextual_analysis")
    nonempty(value.get("reasoning"), "reasoning")
    entities = value.get("knowledge_entities", None if require_entities else [])
    if not isinstance(entities, list) or any(not isinstance(e, str) or not e.strip() for e in entities):
        raise ValueError("knowledge_entities must be a list of nonempty strings")
    value["knowledge_entities"] = entities
    return value


def appendix_verdict(text):
    value = json_object(text)
    if type(value.get("label")) is not int or value["label"] not in (0, 1):
        raise ValueError("label must be integer 0 or 1")
    tags = re.findall(r"(?:<<LABEL>>|«LABEL»)\s*(\S+)", text)
    if tags and (len(tags) != 1 or tags[0] not in ("0", "1") or int(tags[0]) != value["label"]):
        raise ValueError("Conflicting LABEL tag and JSON label")
    return {"prediction": value["label"], "confidence": probability(value.get("conf"), "conf"),
            "p_sarcasm": None, "explanation": nonempty(value.get("reasoning"), "reasoning")}


def maintext_verdict(text):
    value = json_object(text)
    p = probability(value.get("p_sarcasm"), "p_sarcasm")
    label = int(p >= 0.5)
    return {"prediction": label, "confidence": p if label else 1 - p, "p_sarcasm": p,
            "explanation": nonempty(value.get("explanation"), "explanation")}
