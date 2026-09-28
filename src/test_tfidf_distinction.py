import sys
sys.path.insert(0, 'src')
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

# Test 1: Williams & Pacheco (True match)
names_true = [
    "Williams and Pacheco Detailing LLC",
    "Williams and Pacheco LLC Partners"
]
vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
X_t = vec.fit_transform(names_true)
sim_true = (X_t[0] * X_t[1].T).toarray()[0, 0]
print("True Match (Williams & Pacheco Detailing vs Partners):")
print(f"  TF-IDF Cosine Similarity: {sim_true:.4f}")

# Test 2: Spartans India vs Spartans Wagon (False match)
names_false = [
    "Spartans India Private Limited",
    "Spartans Wagon Private Limited"
]
vec2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
X_f = vec2.fit_transform(names_false)
sim_false = (X_f[0] * X_f[1].T).toarray()[0, 0]
print("\nFalse Match (Spartans India vs Spartans Wagon):")
print(f"  TF-IDF Cosine Similarity: {sim_false:.4f}")

# Test 3: Character n-gram TF-IDF (robust to typos)
vec3 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
X_t3 = vec3.fit_transform(names_true)
sim_char_true = (X_t3[0] * X_t3[1].T).toarray()[0, 0]

X_f3 = vec3.fit_transform(names_false)
sim_char_false = (X_f3[0] * X_f3[1].T).toarray()[0, 0]

print("\nCharacter n-gram TF-IDF (3-5 chars):")
print(f"  True Match Sim:  {sim_char_true:.4f}")
print(f"  False Match Sim: {sim_char_false:.4f}")
