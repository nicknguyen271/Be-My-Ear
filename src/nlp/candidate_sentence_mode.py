
from itertools import product
import math
import re
import numpy as np

GLOSS_NORMALISATION = {
    "me": "I", "i": "I", "I": "I", "my": "my", "mine": "my", "myself": "I",
    "you": "you", "your": "your",
    "want": "want", "need": "need", "like": "like", "ask": "ask", "help": "help",
    "water": "water", "drink": "drink", "cup": "cup", "milk": "milk", "coffee": "coffee",
    "hospital": "hospital", "doctor": "doctor", "pain": "pain", "stomach": "stomach",
    "abdomen": "stomach", "sick": "sick", "emergency": "emergency",
    "bathroom": "bathroom", "toilet": "bathroom",
    "food": "food", "eat": "eat", "hungry": "hungry",
    "home": "home", "go": "go", "school": "school", "work": "work",
    "yes": "yes", "no": "no", "please": "please",
    "thank": "thank you", "thanks": "thank you", "sorry": "sorry",
}

SEMANTIC_CATEGORY = {
    "I": "person", "you": "person", "my": "person",
    "want": "intent", "need": "intent", "like": "intent", "ask": "intent",
    "help": "help",
    "water": "object", "drink": "object", "cup": "object",
    "milk": "object", "coffee": "object", "food": "object",
    "eat": "action", "go": "action",
    "hospital": "place", "doctor": "person_medical",
    "bathroom": "place", "home": "place", "school": "place", "work": "place",
    "pain": "medical", "stomach": "body", "sick": "medical", "emergency": "emergency",
    "yes": "response", "no": "response",
    "please": "politeness", "thank you": "politeness", "sorry": "politeness",
}

def normalise_gloss(gloss):
    gloss = str(gloss).strip()
    return GLOSS_NORMALISATION.get(gloss.lower(), gloss)

def get_category(gloss):
    return SEMANTIC_CATEGORY.get(gloss, "unknown")

def get_top_candidates(event, max_candidates_per_event=3):
    candidates = []
    for item in event["top_k"][:max_candidates_per_event]:
        normalised = normalise_gloss(item["gloss"])
        candidates.append({
            "raw_gloss": item["gloss"],
            "gloss": normalised,
            "probability": float(item["probability"]),
            "category": get_category(normalised),
            "event_id": event["event_id"],
            "status": event.get("status", "unknown"),
        })
    return candidates

def generate_candidate_sequences(events, max_candidates_per_event=3):
    candidate_lists = [get_top_candidates(e, max_candidates_per_event) for e in events]
    return [list(seq) for seq in product(*candidate_lists)]

def probability_score(sequence):
    probs = [max(item["probability"], 1e-6) for item in sequence]
    return sum(math.log(p) for p in probs) / max(len(probs), 1)

def status_bonus(sequence):
    bonus = 0.0
    for item in sequence:
        if item["status"] == "accepted":
            bonus += 0.25
        elif item["status"] == "locked":
            bonus += 0.18
        elif item["status"] == "uncertain":
            bonus += 0.05
    return bonus

def grammar_pattern_score(glosses, categories):
    score = 0.0
    if len(categories) >= 3:
        if categories[0] == "person" and categories[1] == "intent" and categories[2] in ["object", "place"]:
            score += 1.2
    if len(categories) >= 2:
        if categories[0] == "intent" and categories[1] in ["object", "place"]:
            score += 0.8
    if "pain" in glosses and ("stomach" in glosses or "body" in categories):
        score += 1.0
    if "help" in glosses and any(c in ["medical", "place", "body"] for c in categories):
        score += 0.9
    for i in range(len(categories) - 1):
        if glosses[i] == "go" and categories[i + 1] == "place":
            score += 0.8
    if "please" in glosses or "thank you" in glosses:
        score += 0.15
    return score

def semantic_usefulness_score(glosses, categories):
    score = 0.0
    useful = {"person", "intent", "object", "place", "medical", "body", "help", "action", "emergency"}
    score += sum(0.1 for c in categories if c in useful)
    if any(g in ["emergency", "help", "hospital", "doctor", "pain", "sick"] for g in glosses):
        score += 0.5
    return score

def score_sequence(sequence):
    glosses = [item["gloss"] for item in sequence]
    categories = [item["category"] for item in sequence]
    p = probability_score(sequence)
    g = grammar_pattern_score(glosses, categories)
    s = semantic_usefulness_score(glosses, categories)
    b = status_bonus(sequence)
    return {
        "glosses": glosses,
        "categories": categories,
        "probability_score": p,
        "grammar_score": g,
        "semantic_score": s,
        "status_bonus": b,
        "total_score": p + g + s + b,
    }

def remove_repeated_words(words):
    cleaned = []
    for word in words:
        if len(cleaned) == 0 or cleaned[-1] != word:
            cleaned.append(word)
    return cleaned

def build_natural_sentence(glosses):
    glosses = [normalise_gloss(g) for g in glosses]
    glosses = remove_repeated_words(glosses)
    gloss_set = set(glosses)

    if "emergency" in gloss_set:
        return "This is an emergency."
    if "help" in gloss_set and "hospital" in gloss_set:
        return "I need help. I need to go to the hospital."
    if "pain" in gloss_set and "stomach" in gloss_set:
        return "I have stomach pain."
    if "sick" in gloss_set and "doctor" in gloss_set:
        return "I am sick. I need a doctor."
    if "go" in gloss_set and "hospital" in gloss_set:
        return "I need to go to the hospital."
    if "bathroom" in gloss_set:
        if "need" in gloss_set or "want" in gloss_set:
            return "I need to use the bathroom."
        return "Bathroom."

    subject, intent, obj, politeness = None, None, None, None
    for g in glosses:
        c = get_category(g)
        if subject is None and c == "person":
            subject = g
        if intent is None and c == "intent":
            intent = g
        if obj is None and c in ["object", "place"]:
            obj = g
        if c == "politeness":
            politeness = g

    if subject is None and intent is not None:
        subject = "I"

    if subject is not None and intent is not None and obj is not None:
        sentence = f"{subject} {intent} {obj}"
        if politeness == "please":
            sentence += ", please"
        return sentence[0].upper() + sentence[1:] + "."

    if intent is not None and obj is not None:
        sentence = f"I {intent} {obj}"
        if politeness == "please":
            sentence += ", please"
        return sentence[0].upper() + sentence[1:] + "."

    if "hungry" in gloss_set:
        return "I am hungry."
    if "water" in gloss_set or "drink" in gloss_set:
        return "I want a drink."
    if "thank you" in gloss_set:
        return "Thank you."
    if "sorry" in gloss_set:
        return "I am sorry."
    if len(glosses) == 1:
        return glosses[0].capitalize() + "."

    sentence = " ".join(glosses)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence[0].upper() + sentence[1:] + "."

def sentence_confidence_label(best_item, scored_sequences):
    sorted_scores = sorted([item["total_score"] for item in scored_sequences], reverse=True)
    best = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else best - 1.0
    gap = best - second
    if best_item["grammar_score"] >= 1.0 and gap >= 0.15:
        return "High", gap
    if best_item["grammar_score"] >= 0.8 or best_item["semantic_score"] >= 0.5:
        return "Medium", gap
    return "Low", gap

def candidate_sentence_mode(events, max_candidates_per_event=3):
    sequences = generate_candidate_sequences(events, max_candidates_per_event)
    scored = [{"sequence": seq, **score_sequence(seq)} for seq in sequences]
    best = max(scored, key=lambda x: x["total_score"])
    sentence = build_natural_sentence(best["glosses"])
    confidence, gap = sentence_confidence_label(best, scored)
    alternatives = sorted(scored, key=lambda x: x["total_score"], reverse=True)[:5]
    return {
        "sentence": sentence,
        "confidence": confidence,
        "score_gap": float(gap),
        "selected_glosses": best["glosses"],
        "selected_categories": best["categories"],
        "best_score": float(best["total_score"]),
        "alternatives": [
            {
                "glosses": alt["glosses"],
                "sentence": build_natural_sentence(alt["glosses"]),
                "score": float(alt["total_score"]),
            }
            for alt in alternatives
        ],
    }
