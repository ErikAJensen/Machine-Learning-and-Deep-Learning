import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# 1) Data (nå med tall istedenfor strings for type)
# type: shotgun=0, rifle=1
data = [
    ["Specter 1882", 0, 200, "close"],   # shotgun
    ["Romero 77",    0, 200, "close"],   # shotgun
    ["Winfield M1873",1, 110, "mid"],    # rifle
    ["Sparks LRR",   1, 149, "long"],    # rifle
]

df = pd.DataFrame(data, columns=["weapon","type","damage","playstyle"])

X = df[["type","damage"]]   # input = type og damage
y = df["playstyle"]         # output = fasit

# 2) Tren modell
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X, y)

# 3) La modellen gjette
print("Fasit :", list(y))
print("Gjett :", list(model.predict(X)))

# 4) Se reglene den lærte
rules = export_text(model, feature_names=["type","damage"])
print("\nRegler:\n", rules)
