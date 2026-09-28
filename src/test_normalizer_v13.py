import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
from normalizer import normalize_business_name, normalize_address, extract_structured_fields

print("=" * 60)
print("NORMALIZER V13 VERIFICATION")
print("=" * 60)

# Test 1: Unidecode transliteration
print("\n--- Test: Indic Script Transliteration ---")
tests = [
    ("Pioneer Indian Trading", "pioneer indian trading"),
    ("\u092a\u093e\u092f\u094b\u0928\u093f\u092f\u0930 \u0907\u0902\u0921\u093f\u092f\u0928 \u091f\u094d\u0930\u0947\u0921\u093f\u0902\u0917", "payoniyar indiyan treding"),
    ("\u09aa\u09be\u09af\u09bc\u09cb\u09a8\u09bf\u09af\u09bc\u09b0", "Bengali pioneer"),
    ("Pvt Super Logistics Ltd", "super logistics private limited"),
    ("D/B/A Quick Stop", "quick stop"),
    ("#00381 Main St", "381 Main St - leading zeros stripped"),
]

for raw, note in tests:
    norm = normalize_business_name(raw)
    print(f"  IN:   {raw}")
    print(f"  OUT:  {norm}")
    print(f"  NOTE: {note}")
    print()

# Test 2: House number leading zero stripping
print("--- Test: House Number Leading Zero Stripping ---")
cases = [
    ("00381 Main Street, New York, NY 10001", "381"),
    ("381 Main Street, New York, NY 10001", "381"),
    ("001 Oak Ave, Dallas, TX 75001", "1"),
    ("#0042 Elm St, Chicago, IL 60601", "42"),
]
for addr, expected in cases:
    r = extract_structured_fields(addr)
    status = "OK" if r["house_number"] == expected else f"FAIL (got {r['house_number']})"
    print(f"  {addr[:40]:<40} house={r['house_number']:<8} expected={expected:<6} [{status}]")

# Test 3: Legal token sorting
print("\n--- Test: Legal Token Ordering ---")
pairs = [
    ("Pvt Super Logistics Ltd", "Super Logistics Pvt Ltd"),
    ("Private Alpha Corp", "Alpha Corp Private"),
]
for a, b in pairs:
    na = normalize_business_name(a)
    nb = normalize_business_name(b)
    match = "MATCH" if na == nb else "DIFFER"
    print(f"  A: {a:<30} -> {na}")
    print(f"  B: {b:<30} -> {nb}")
    print(f"  [{match}]")
    print()

# Test 4: Address with Indic scripts
print("--- Test: Address Indic Script Transliteration ---")
addrs = [
    "#917 TULSHIBERIYA ULUBERIA, HOWRAH, \u09aa\u09b6\u09cd\u099a\u09bf\u09ae\u09ac\u0999\u09cd\u0997 (India)",
    "Tulshiberiya Uluberia, Howrah, West Bengal (India)",
]
for a in addrs:
    r = extract_structured_fields(a)
    print(f"  IN:   {a}")
    print(f"  NORM: {r['address_normalized']}")
    print(f"  HOUSE: {r['house_number']} | PIN: {r['postal_code']}")
    print()

print("NORMALIZER V13 VERIFICATION COMPLETE")
