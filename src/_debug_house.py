import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
from normalizer import normalize_address, extract_structured_fields

r = extract_structured_fields('#0042 Elm St, Chicago, IL 60601')
print(f"norm={r['address_normalized']}")
print(f"house={r['house_number']}")
print(f"postal={r['postal_code']}")
