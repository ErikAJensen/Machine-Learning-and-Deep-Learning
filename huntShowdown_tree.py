# hunt_tree_clean.py
# Mål: Lett å forstå beslutningstre
# - Klare mønstre + noen få unntak
# - Enkle regler som gir mening
# - Tekstlige regler + enkel evaluering

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -----------------------------
# 1) Datasett (enkelt og ryddig)
# playstyle-regel (intuitiv):
#   - ammo == 'shotgun'  → close
#   - ammo == 'long'     → long
#   - ellers (small/medium) → mid
# Unntak (for realisme):
#   - "Sparks Sil." (long + silenced) → mid
#   - "Winfield HighRPM" (rifle small + høy rpm) → close
# -----------------------------
data = [
# weapon,             type,     ammo,     silenced, price, damage, rpm,  playstyle
["Specter 1882",      "shotgun","shotgun","no",     330,   200,    60,   "close"],
["Romero 77",         "shotgun","shotgun","no",     66,    200,    30,   "close"],
["Rival 78",          "shotgun","shotgun","no",     250,   190,    75,   "close"],
["Romero Alamo",       "shotgun","shotgun","no",     500,   200,    100,    "close"],

["Winfield M1873",    "rifle",  "small",  "no",     201,   110,    120,  "mid"],
["Winfield Sil.",     "rifle",  "small",  "yes",    260,   105,    120,  "mid"],
["Vetterli 71",       "rifle",  "medium", "no",     333,   130,    50,   "mid"],
["Springfield",       "rifle",  "medium", "no",     150,   125,    45,   "mid"],

["Sparks LRR",        "rifle",  "long",   "no",     130,   149,    16,   "long"],
["Mosin-Nagant",      "rifle",  "long",   "no",     490,   136,    30,   "long"],
["Uppercut",          "pistol", "long",   "no",     275,   130,    60,   "long"],

# --- UNNTAK ---
["Sparks Sil.",       "rifle",  "long",   "yes",    200,   140,    16,   "mid"],   # egentlig long, men vi sier mid
["Winfield HighRPM",  "rifle",  "small",  "no",     230,   108,    160,  "close"], # egentlig mid, men vi sier close

# Noen ekstra for balanse
["Nagant Silencer",   "pistol", "small",  "yes",     96,    91,    120,  "mid"],
["Caldwell Pax",      "pistol", "medium", "no",     100,   110,     90,  "mid"],
["Crown & King",      "shotgun","shotgun","no",     350,   200,     70,  "close"],
]

df = pd.DataFrame(data, columns=["weapon","type","ammo","silenced","price","damage","rpm","playstyle"])

# Features og label
X = df[["weapon","type","ammo","silenced","price","damage","rpm"]]
y = df["playstyle"]

cat_cols = ["weapon","type","ammo","silenced"]
num_cols = ["price","damage","rpm"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Enkelt tre – lav dybde for lesbarhet
tree = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=42)
pipe = Pipeline([("prep", preprocess), ("tree", tree)])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y, random_state=42
)

# Tren
pipe.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# 1) Totalt riktig/feil på testsettet
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
tot = len(y_test)
riktig = (y_pred == y_test).sum()
feil = tot - riktig

print(f"\n✅ Totalt riktig: {riktig}/{tot}  ({acc*100:.1f}%)")
print(f"❌ Totalt feil  : {feil}/{tot}  ({(1-acc)*100:.1f}%)")


# 2) List opp hvilke rader som ble feil – med våpennavn
print("\nFeilklassifiserte eksempler:")
for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
    if yt != yp:
        wpn = X_test.iloc[i]["weapon"]   # X_test er en DataFrame, har 'weapon'-kolonnen
        print(f"- {wpn:20s}  fasit: {yt:5s}  →  modell: {yp:5s}")

# 3) Per-klasse telling (hvor mange riktig per klasse)
from collections import Counter
support_per_class = Counter(y_test)
riktig_per_class = Counter()
for yt, yp in zip(y_test, y_pred):
    if yt == yp:
        riktig_per_class[yt] += 1

print("\nRiktig per klasse:")
for cls in pipe.classes_:
    sup = support_per_class.get(cls, 0)
    kor = riktig_per_class.get(cls, 0)
    pct = (kor / sup) if sup else 0.0
    print(f"- {cls:5s}: {kor}/{sup}  ({pct:.2%})")

# Eval
# Kommenterer bort disse for enklere oversikt over resultatente 
# print("Accuracy (train):", round(pipe.score(X_train, y_train), 2))
# print("Accuracy (test) :", round(pipe.score(X_test, y_test), 2))
# print("\nClassification report:\n", classification_report(y_test, pipe.predict(X_test)))

# Regler i lesbar form (one-hot → klartekst)
ohe = pipe.named_steps["prep"].named_transformers_["cat"]
feat_names = list(ohe.get_feature_names_out(cat_cols)) + num_cols
rules_raw = export_text(pipe.named_steps["tree"], feature_names=feat_names)

def pretty_rules(rules: str) -> str:
    lines = []
    for ln in rules.splitlines():
        line = ln
        for col in cat_cols:
            pref = col + "_"
            if pref in line:
                idx = line.find(pref)
                rest = line[idx+len(pref):]
                val = rest.split(" ")[0]
                if "<= 0.50" in line:
                    line = line.replace(f"{pref}{val} <= 0.50", f"{col} != '{val}'")
                if ">  0.50" in line:
                    line = line.replace(f"{pref}{val} >  0.50", f"{col} == '{val}'")
        lines.append(line)
    return "\n".join(lines).replace("|---", "├─").replace("|   ", "│  ")

print("\nRegler (ryddet):\n")
print(pretty_rules(rules_raw))

# Noen samples for å SE hva den sier + sannsynligheter
samples = pd.DataFrame([
    {"weapon":"Specter 1882", "type":"shotgun","ammo":"shotgun","silenced":"no","price":330,"damage":200,"rpm":60},
    {"weapon":"Winfield M1873","type":"rifle","ammo":"small","silenced":"no","price":201,"damage":110,"rpm":120},
    {"weapon":"Sparks LRR",   "type":"rifle","ammo":"long","silenced":"no","price":130,"damage":149,"rpm":16},
    {"weapon":"Sparks Sil.",  "type":"rifle","ammo":"long","silenced":"yes","price":200,"damage":140,"rpm":16}, # unntak
    {"weapon":"Winfield HighRPM","type":"rifle","ammo":"small","silenced":"no","price":230,"damage":108,"rpm":160}, # unntak
    {"weapon":"Dolch Drilling",    "type":"shotgun","ammo":"shotgun","silenced":"no","price":380,"damage":190,"rpm":70}
])
pred = pipe.predict(samples)
proba = pipe.predict_proba(samples)

print("\nPrediksjoner på samples:")
for i, row in samples.iterrows():
    probs = dict(zip(pipe.classes_, np.round(proba[i], 2)))
    print(f"- {row['weapon']:<18} → pred: {pred[i]:5s}  probs: {probs}")

# Confusion matrix (for oversikt)
cm = confusion_matrix(y_test, pipe.predict(X_test), labels=pipe.classes_)
cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in pipe.classes_],
                        columns=[f"pred_{c}" for c in pipe.classes_])
print("\nConfusion matrix:\n", cm_df)
