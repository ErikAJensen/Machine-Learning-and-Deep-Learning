import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ANSI farger til terminal
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

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
riktig_ant = sum(y_pred == y_test)
feil_ant = totalt - riktig_ant
riktig_pct = acc * 100
feil_pct = (1 - acc) * 100

print("\n=== Modellresultater ===")
print(f"Totalt antall studenter i testsettet: {totalt}")
print(GREEN + f"Riktig: {riktig_ant} av {totalt} ({riktig_pct:.2f}%)" + RESET)
print(RED   + f"Feil:  {feil_ant} av {totalt} ({feil_pct:.2f}%)" + RESET)
print("\nConfusion Matrix:\n", cm)
print("\nKlasserapport:\n", classification_report(y_test, y_pred))

# === 6) Vis beslutningsregler ===
print("\n=== Beslutningsregler (teknisk) ===")
print(export_text(model, feature_names=list(X.columns)))

print("\n=== Tolkning i enkelt språk ===")
print("Modellen forutsier stressnivå (0=Lavt, 1=Medium, 2=Høyt) basert på variabler som blood_pressure,")
print("social_support, sleep_quality, bullying, osv. Reglene over viser hvilke grenser den bruker.")

# === 7) Lagre prediksjoner til CSV ===
out = X_test.copy()
out["actual"] = y_test.values
out["predicted"] = y_pred
out.to_csv("prediksjoner_tree.csv", index=False)
print("\nPrediksjoner lagret i: prediksjoner_tree.csv")

# === 8) Lesbar tabell i terminalen ===
label_map = {0: "Lavt stress", 1: "Medium stress", 2: "Høyt stress"}
def as_label(v):
    try:
        return label_map[int(v)]
    except Exception:
        return str(v)

out["actual_label"] = out["actual"].apply(as_label)
out["predicted_label"] = out["predicted"].apply(as_label)

print("\n=== De 10 første studentene i testsettet (lett å lese) ===")
print(out[["actual_label", "predicted_label"]].head(10).to_string(index=False))

# Lagre full tabell til TXT
out_txt = "prediksjoner_lesbart.txt"
out[["actual_label", "predicted_label"]].to_string(open(out_txt, "w", encoding="utf-8"), index=False)
print(f"\nLesbar tabell lagret i: {out_txt}")

# === 9) Viktigste variabler ===
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Viktigste variabler (feature importance) ===")
print(imp.head(10))

# === 10) Tegn og lagre treet ===
plt.figure(figsize=(20, 10))
class_names = [str(c) for c in model.classes_]
plot_tree(model, feature_names=X.columns, class_names=class_names, filled=True, rounded=True)
plt.tight_layout()
plt.savefig("tree.png", dpi=200, bbox_inches="tight")
print("Tre-bilde lagret i: tree.png")
plt.show()
