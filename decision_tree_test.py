from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

# 1. Last inn datasettet
iris = load_iris()
X, y = iris.data, iris.target

# 2. Tren treet
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# 3. Print treet med labels
tree_rules = export_text(
    model,
    feature_names=list(iris.feature_names),
    show_weights=True
)

# Bytt ut class-numrene med navn
for i, name in enumerate(iris.target_names):
    tree_rules = tree_rules.replace(f"class: {i}", f"class: {name}")

print(tree_rules)

# 4. Test en prediksjon
ny_blomst = np.array([[5.0, 3.4, 1.6, 0.2]])
pred = model.predict(ny_blomst)[0]
print("Prediksjon:", iris.target_names[pred])
