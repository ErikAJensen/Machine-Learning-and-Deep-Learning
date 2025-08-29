import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os, sys

# === Farger (kun hvis terminalen støtter det) ===
supports_color = sys.stdout.isatty() and (os.name != "nt" or "WT_SESSION" in os.environ or "TERM_PROGRAM" in os.environ)
GREEN = "\033[92m" if supports_color else ""
RED   = "\033[91m" if supports_color else ""
RESET = "\033[0m"  if supports_color else ""

# === 1) Les inn datasett ===
CSV = "StressLevelDataset.csv"
df = pd.read_csv(CSV)

print("Kolonner:", df.columns.tolist())
print(df.head())

# === 2) X (input) og y (mål) ===
y = df["stress_level"]
X = df.drop(columns=["stress_level"])

# === 3) Train/Test-splitt ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# === 4) Tren Decision Tree ===
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# === 5) Evaluer på test ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

totalt = len(y_test)
riktig_ant = int((y_pred == y_test).sum())
feil_ant = totalt - riktig_ant
riktig_pct = 100 * riktig_ant / totalt
feil_pct = 100 - riktig_pct

print("\n=== Modellresultater ===")
print(f"Totalt antall studenter i testsettet: {totalt}")
print(GREEN + f"Riktig: {riktig_ant} av {totalt} ({riktig_pct:.2f}%)" + RESET)
print(RED   + f"Feil:  {feil_ant} av {totalt} ({feil_pct:.2f}%)" + RESET)
print("\nConfusion Matrix:\n", cm)
print("\nKlasserapport:\n", classification_report(y_test, y_pred))

# === 6) Beslutningsregler ===
print("\n=== Beslutningsregler (teknisk) ===")
print(export_text(model, feature_names=list(X.columns)))

# === 7) Lagre prediksjoner ===
out = X_test.copy()
out["actual"] = y_test.values
out["predicted"] = y_pred
out.to_csv("prediksjoner_tree.csv", index=False)
print("\nPrediksjoner lagret i: prediksjoner_tree.csv")

# === 8) Lesbar tabell og analyse ===
label_map = {0: "Lavt stress", 1: "Medium stress", 2: "Høyt stress"}
def as_label(v):
    try:
        return label_map[int(v)]
    except Exception:
        return str(v)

out["actual_label"] = out["actual"].apply(as_label)
out["predicted_label"] = out["predicted"].apply(as_label)

print("\n=== De 10 første studentene (Fasit vs Modell) ===")
print(out[["actual_label", "predicted_label"]].head(10).to_string(index=False))

# Per-klasse nøyaktighet
print("\nPer-klasse treffsikkerhet:")
for c in np.unique(y_test):
    mask = (y_test == c)
    treff = (y_pred[mask] == y_test[mask]).mean()
    navn = label_map.get(int(c), str(c))
    print(f"  {navn}: {treff*100:.2f}% ( {mask.sum()} studenter )")

# Feilklassifiserte eksempler
mis = out[out["actual"] != out["predicted"]].copy()
cols_to_show = ["actual_label","predicted_label","blood_pressure","sleep_quality","social_support","bullying"]
exist_cols = [c for c in cols_to_show if c in mis.columns]
print("\n=== Feilklassifiserte eksempler (inntil 10) ===")
if mis.empty:
    print(GREEN + "Ingen feilklassifiseringer i testsettet 🎉" + RESET)
else:
    print(RED + mis[exist_cols].head(10).to_string(index=False) + RESET)

# === 9) Lagre en rapport til TXT ===
report_txt = [
    "STRESS MODELL – RAPPORT",
    f"Totalt testsett: {totalt}",
    f"Riktig: {riktig_ant} ({riktig_pct:.2f}%)",
    f"Feil:  {feil_ant} ({feil_pct:.2f}%)",
    "Per-klasse treffsikkerhet:"
]
for c in np.unique(y_test):
    mask = (y_test == c)
    treff = (y_pred[mask] == y_test[mask]).mean()
    navn = label_map.get(int(c), str(c))
    report_txt.append(f"  {navn}: {treff*100:.2f}% (n={mask.sum()})")

report_txt.append("")
report_txt.append("De 10 første (Fasit vs Modell):")
report_txt.append(out[["actual_label","predicted_label"]].head(10).to_string(index=False))

with open("rapport_stress.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_txt))
print("\nRapport lagret i: rapport_stress.txt")

# === 10) Viktigste variabler ===
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Viktigste variabler (feature importance) ===")
print(imp.head(10))

# === 11) Tegn og lagre treet ===
plt.figure(figsize=(20, 10))
class_names = [str(c) for c in model.classes_]
plot_tree(model, feature_names=X.columns, class_names=class_names, filled=True, rounded=True)
plt.tight_layout()
plt.savefig("tree.png", dpi=200, bbox_inches="tight")
print("Tre-bilde lagret i: tree.png")
plt.show()
