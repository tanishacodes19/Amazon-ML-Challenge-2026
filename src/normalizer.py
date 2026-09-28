import re
import unicodedata
from unidecode import unidecode

# Legal Suffix Dictionary
LEGAL_MAP = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\binc\b": "incorporated",
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\bdept\b": "department",
    r"\bassoc\b": "association",
    r"\bintl\b": "international",
    r"\bmfg\b": "manufacturing",
    r"\btech\b": "technology",
    r"\bsvcs\b": "services",
    r"\bmgmt\b": "management",
    # French corporate forms
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\bsasu\b": "sasu",
    r"\bsa\b": "sa",
    r"\beurl\b": "eurl",
    r"\bsnc\b": "snc",
    r"\bgie\b": "gie",
    r"\bste\b": "societe",
}

# Address Abbreviations
ADDR_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bblvd\b": "boulevard",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bbldg\b": "building",
    r"\bfl\b": "floor",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    # French address tokens
    r"\brue\b": "street",
    r"\bbd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\ball\b": "allee",
    r"\ballee\b": "allee",
    r"\bchemin\b": "chemin",
    r"\bch\b": "chemin",
    r"\bimpasse\b": "impasse",
    r"\bimp\b": "impasse",
    r"\bcedex\b": "cedex",
}

# State & Region Codes (US & India)
STATE_MAP = {
    # US
    r"\bca\b": "california",
    r"\bny\b": "new york",
    r"\btx\b": "texas",
    r"\bfl\b": "florida",
    r"\bil\b": "illinois",
    r"\bpa\b": "pennsylvania",
    r"\boh\b": "ohio",
    r"\bga\b": "georgia",
    r"\bnc\b": "north carolina",
    r"\bmi\b": "michigan",
    r"\bnj\b": "new jersey",
    r"\bva\b": "virginia",
    r"\bwa\b": "washington",
    r"\baz\b": "arizona",
    r"\bma\b": "massachusetts",
    r"\bwi\b": "wisconsin",
    r"\bco\b": "colorado",
    r"\bmn\b": "minnesota",
    r"\bmo\b": "missouri",
    r"\bmd\b": "maryland",
    r"\bin\b": "indiana",
    r"\btn\b": "tennessee",
    
    # India
    r"\bmh\b": "maharashtra",
    r"\bka\b": "karnataka",
    r"\btn\b": "tamil nadu",
    r"\bdl\b": "delhi",
    r"\bup\b": "uttar pradesh",
    r"\bwb\b": "west bengal",
    r"\bgj\b": "gujarat",
    r"\brj\b": "rajasthan",
    r"\bmp\b": "madhya pradesh",
    r"\bap\b": "andhra pradesh",
    r"\bts\b": "telangana",
    r"\bkl\b": "kerala",
    r"\bhr\b": "haryana",
    r"\bpb\b": "punjab",
    r"\bbr\b": "bihar",
    r"\bor\b": "odisha",
}

# Cross-script Indic aliases to English standard
CROSS_SCRIPT_MAP = {
    "পশ্চিমবঙ্গ": "west bengal",
    "ગુજરાત": "gujarat",
    "महाराष्ट्र": "maharashtra",
    "मध्य प्रदेश": "madhya pradesh",
    "मध्यप्रदेश": "madhya pradesh",
    "उत्तर प्रदेश": "uttar pradesh",
    "उत्तरप्रदेश": "uttar pradesh",
    "राजस्थान": "rajasthan",
    "कर्नाटक": "karnataka",
    "तमिलनाडु": "tamil nadu",
    "केरल": "kerala",
    "हरियाणा": "haryana",
    "पंजाब": "punjab",
    "दिल्ली": "delhi",
    "बिहार": "bihar",
    "ओडिशा": "odisha",
}

# Compile patterns for speed
LEGAL_COMPILED = [(re.compile(p), repl) for p, repl in LEGAL_MAP.items()]
ADDR_COMPILED = [(re.compile(p), repl) for p, repl in ADDR_MAP.items()]
STATE_COMPILED = [(re.compile(p), repl) for p, repl in STATE_MAP.items()]

def clean_base(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    # Strip URLs and domains before stripping punctuation
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(?:com|org|net|co|in|us|fr|gov|edu|info|biz)\b", " ", text, flags=re.IGNORECASE)
    # Strip trailing phone numbers and ID tags (e.g. - 3607255560, #44872123)
    text = re.sub(r"(?:-\s*\d{7,}|\b\d{10}\b|#\d{5,})", " ", text)
    # Transliterate ALL non-Latin scripts (Devanagari, Tamil, Bengali, etc.) to Latin
    text = unidecode(text)
    # Strip combining diacritics (e.g. accented Latin: é -> e, ó -> o)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_business_name(text: str) -> str:
    cleaned = clean_base(text)
    if not cleaned:
        return ""
    # Strip honorific prefixes (common in Indian / Asian records)
    cleaned = re.sub(r"^(?:shri|sri|sree|m\s*/\s*s|dr|prof)\s+", "", cleaned)
    # Strip DBA / trading-as prefixes
    cleaned = re.sub(r"^(?:d\s*/?\s*b\s*/?\s*a|dba|t\s*/?\s*a|trading\s+as)\s+", "", cleaned)
    for pat, repl in LEGAL_COMPILED:
        cleaned = pat.sub(repl, cleaned)
    # Move legal suffixes to end for consistent ordering:
    # "private super logistics limited" == "super logistics private limited"
    legal_words = {"private", "limited", "incorporated", "corporation", "company",
                   "llc", "llp", "department", "association", "international",
                   "manufacturing", "technology", "services", "management",
                   "sarl", "sas", "sasu", "sa", "eurl", "snc", "gie", "societe"}
    tokens = cleaned.split()
    core = [t for t in tokens if t not in legal_words]
    legal = [t for t in tokens if t in legal_words]
    cleaned = " ".join(core + legal)
    return re.sub(r"\s+", " ", cleaned).strip()

def normalize_address(text: str) -> str:
    cleaned = clean_base(text)
    if not cleaned:
        return ""
    # Cross-script replacement
    for script_word, eng_word in CROSS_SCRIPT_MAP.items():
        if script_word in cleaned:
            cleaned = cleaned.replace(script_word, eng_word)
    # Address abbreviations
    for pat, repl in ADDR_COMPILED:
        cleaned = pat.sub(repl, cleaned)
    # State abbreviations
    for pat, repl in STATE_COMPILED:
        cleaned = pat.sub(repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

def extract_structured_fields(address_text: str):
    norm = normalize_address(address_text)
    
    # House number: leading digits or digits preceded by # / no / plot
    house_num = ""
    hn_match = re.search(r"(?:^|#|plot\s*no\.?|q\.?no\.?|flat\s*no\.?|unit\s*)\s*(\d+[a-zA-Z]?)", norm)
    if hn_match:
        house_num = hn_match.group(1)
    else:
        for t in norm.split():
            if t and t[0].isdigit():
                house_num = t
                break
    # Strip leading zeros: '00381' -> '381'
    if house_num:
        house_num = house_num.lstrip('0') or '0'
                
    # Postal code: 5 digits (US) or 6 digits (India)
    postal = ""
    pin_match = re.search(r"\b(\d{5,6})\b", norm)
    if pin_match:
        postal = pin_match.group(1)
        
    return {
        "address_normalized": norm,
        "house_number": house_num,
        "postal_code": postal
    }

if __name__ == "__main__":
    test_names = [
        "Pebble Hotel Private Limited",
        "Delta Telecommunication Inc",
        "Delta Tetlecommunication Inc",
        "QRM Solutions Pvt. Ltd.",
        "MONTI'S ONLINE & STORE LLC",
    ]
    test_addrs = [
        "#917 TULSHIBERIYA ULUBERIA, HOWRAH, পশ্চিমবঙ্গ (India)",
        "Tulshiberiya Uluberia, Howrah, West Bengal (India)",
        "1042, Shukrawar Peth Tilak Road, Flat No-1, Pune, MH",
        "27 Tiffany Circle, West Bridgewater, MA 02379",
        "NEW COLONY NO 2. Q.NO 199, BIRLA NAGAR, GWALIOR, M.P, GIRD, मध्य प्रदेश",
    ]
    print("--- Test Name Normalization ---")
    for n in test_names:
        print(f"RAW:  {n}\nNORM: {normalize_business_name(n)}\n")
    print("--- Test Address Normalization ---")
    for a in test_addrs:
        res = extract_structured_fields(a)
        print(f"RAW:    {a}\nNORM:   {res['address_normalized']}\nHOUSE:  {res['house_number']} | PIN: {res['postal_code']}\n")
