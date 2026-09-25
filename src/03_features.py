from typing import Dict, Any
import re
import unicodedata

from rapidfuzz import fuzz


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Normalize business names/addresses while preserving
    useful Unicode characters.

    Steps:
    1. Handle missing values
    2. Unicode normalization
    3. Lowercase
    4. Replace '&' with 'and'
    5. Replace punctuation with spaces
    6. Collapse whitespace
    """

    if value is None:
        return ""

    text = str(value)

    if text.lower() in {"nan", "none", "null"}:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Normalize ampersand
    text = text.replace("&", " and ")

    # Keep Unicode letters/numbers, replace punctuation
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str):
    """Convert normalized text into a set of tokens."""

    if not text:
        return set()

    return set(text.split())


# ============================================================
# JACCARD SIMILARITY
# ============================================================

def jaccard_similarity(text_a: str, text_b: str) -> float:

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# NUMERIC TOKEN EXTRACTION
# ============================================================

def extract_numeric_tokens(text: str):

    if not text:
        return set()

    return set(re.findall(r"\d+", text))


# ============================================================
# NUMERIC TOKEN OVERLAP
# ============================================================

def numeric_overlap(text_a: str, text_b: str) -> float:

    nums_a = extract_numeric_tokens(text_a)
    nums_b = extract_numeric_tokens(text_b)

    if not nums_a and not nums_b:
        return 1.0

    if not nums_a or not nums_b:
        return 0.0

    intersection = len(nums_a & nums_b)
    union = len(nums_a | nums_b)

    return intersection / union if union else 0.0


# ============================================================
# LENGTH DIFFERENCE
# ============================================================

def relative_length_difference(text_a: str, text_b: str) -> float:

    len_a = len(text_a)
    len_b = len(text_b)

    denominator = max(len_a, len_b, 1)

    return abs(len_a - len_b) / denominator


# ============================================================
# NAME FEATURES
# ============================================================

def name_features(name_a: str, name_b: str) -> Dict[str, float]:

    a = normalize_text(name_a)
    b = normalize_text(name_b)

    return {
        "name_exact": float(a == b and a != ""),

        "name_ratio": (
            fuzz.ratio(a, b) / 100.0
            if a and b else 0.0
        ),

        "name_partial_ratio": (
            fuzz.partial_ratio(a, b) / 100.0
            if a and b else 0.0
        ),

        "name_token_sort_ratio": (
            fuzz.token_sort_ratio(a, b) / 100.0
            if a and b else 0.0
        ),

        "name_token_set_ratio": (
            fuzz.token_set_ratio(a, b) / 100.0
            if a and b else 0.0
        ),

        "name_jaccard": jaccard_similarity(a, b),

        "name_length_difference": relative_length_difference(a, b),

        "name_missing": float(not a or not b),
    }


# ============================================================
# ADDRESS FEATURES
# ============================================================

def address_features(address_a: str, address_b: str) -> Dict[str, float]:

    a = normalize_text(address_a)
    b = normalize_text(address_b)

    nums_a = extract_numeric_tokens(a)
    nums_b = extract_numeric_tokens(b)

    return {
        "address_exact": float(a == b and a != ""),

        "address_ratio": (
            fuzz.ratio(a, b) / 100.0
            if a and b else 0.0
        ),

        "address_partial_ratio": (
            fuzz.partial_ratio(a, b) / 100.0
            if a and b else 0.0
        ),

        "address_token_sort_ratio": (
            fuzz.token_sort_ratio(a, b) / 100.0
            if a and b else 0.0
        ),

        "address_token_set_ratio": (
            fuzz.token_set_ratio(a, b) / 100.0
            if a and b else 0.0
        ),

        "address_jaccard": jaccard_similarity(a, b),

        "address_numeric_overlap": numeric_overlap(a, b),

        "address_length_difference": relative_length_difference(a, b),

        "address_missing": float(not a or not b),

        "address_number_count_a": float(len(nums_a)),

        "address_number_count_b": float(len(nums_b)),
    }


# ============================================================
# COMPLETE PAIR FEATURE VECTOR
# ============================================================

def pair_features(record_a: Dict[str, Any],
                  record_b: Dict[str, Any]) -> Dict[str, float]:

    features = {}

    # Name features
    features.update(
        name_features(
            record_a.get("business_name"),
            record_b.get("business_name")
        )
    )

    # Address features
    features.update(
        address_features(
            record_a.get("business_address"),
            record_b.get("business_address")
        )
    )

    # Country feature
    country_a = normalize_text(record_a.get("country"))
    country_b = normalize_text(record_b.get("country"))

    features["country_match"] = float(
        country_a != "" and
        country_b != "" and
        country_a == country_b
    )

    features["country_missing"] = float(
        not country_a or not country_b
    )

    # Source information
    id_a = str(record_a.get("entity_id", ""))
    id_b = str(record_b.get("entity_id", ""))

    features["source2_candidate"] = float(
        id_b.startswith("S2-")
    )

    features["source3_candidate"] = float(
        id_b.startswith("S3-")
    )

    return features


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    record_1 = {
        "entity_id": "S1-TEST",
        "business_name": "Prime Money",
        "business_address": "17560 Ellis Road, Tahlequah, OK",
        "country": "US",
    }

    record_2 = {
        "entity_id": "S2-TEST",
        "business_name": "Prime Money LLC",
        "business_address": "17560 Ellis Rd, Tahlequah, Oklahoma",
        "country": "US",
    }

    features = pair_features(record_1, record_2)

    print("\nPAIR FEATURE TEST")
    print("=" * 60)

    for feature_name, value in features.items():
        print(f"{feature_name:35s}: {value}")