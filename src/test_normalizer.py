import sys
sys.stdout.reconfigure(encoding='utf-8')
from normalizer import normalize_business_name, normalize_address, extract_structured_fields

def test_norm():
    cases_name = [
        ("Pebble Hotel Private Limited", "pebble hotel private limited"),
        ("Delta Telecommunication Inc", "delta telecommunication incorporated"),
        ("Delta Tetlecommunication Inc", "delta tetlecommunication incorporated"),
        ("QRM Solutions Pvt. Ltd.", "qrm solutions private limited"),
        ("MONTI'S ONLINE & STORE LLC", "montis online and store llc"),
        ("Tata Motors Ltd", "tata motors limited"),
        ("Reliance Industries Co.", "reliance industries company"),
    ]
    print("Testing Name Normalizer:")
    for raw, expected in cases_name:
        out = normalize_business_name(raw)
        print(f"  {raw!r} -> {out!r}")
        
    cases_addr = [
        ("#917 TULSHIBERIYA ULUBERIA, HOWRAH, পশ্চিমবঙ্গ (India)", "917 tulshiberiya uluberia howrah west bengal india"),
        ("Tulshiberiya Uluberia, Howrah, West Bengal (India)", "tulshiberiya uluberia howrah west bengal india"),
        ("1042, Shukrawar Peth Tilak Road, Flat No-1, Pune, MH", "1042 shukrawar peth tilak road flat no 1 pune maharashtra"),
        ("27 Tiffany Circle, West Bridgewater, MA 02379", "27 tiffany circle west bridgewater massachusetts 02379"),
        ("WALERGA ROAD, CA, ANTELOPE", "walerga road california antelope"),
    ]
    print("\nTesting Address Normalizer & Structured Fields:")
    for raw, _ in cases_addr:
        res = extract_structured_fields(raw)
        print(f"  {raw!r} -> {res['address_normalized']!r}")
        print(f"    House: {res['house_number']!r}, PIN: {res['postal_code']!r}")

if __name__ == "__main__":
    test_norm()
