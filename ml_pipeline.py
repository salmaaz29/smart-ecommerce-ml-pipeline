import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("dataset_features.csv", engine="python")
print(f"Dataset : {len(df)} produits\n")

# ============================================================
# MODULE 1 — CLUSTERING KMEANS CORRIGÉ (K=5)
# La courbe du coude montre clairement K=5
# ============================================================
print("=" * 55)
print("MODULE 1 — CLUSTERING KMEANS (K=5)")
print("=" * 55)

features_cluster = ["log_prix", "promo_bin", "stock_bin", "categorie_id", "shop_id"]
X_cluster = df[features_cluster].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K=5 — meilleur coude visible sur ta courbe
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Silhouette Score — mesure la qualité des clusters (entre -1 et 1, plus c'est haut mieux c'est)
sil = silhouette_score(X_scaled, df["cluster"])
print(f"Silhouette Score (K=5) : {sil:.3f}")

# Profil de chaque cluster
profils = df.groupby("cluster").agg(
    prix_moyen=("prix", "mean"),
    score_moyen=("score_popularite", "mean"),
    promo_pct=("promo_bin", "mean"),
    nb_produits=("produit", "count")
).round(2)
print("\nProfil des clusters :")
print(profils.to_string())

# Nommage automatique des clusters
def nommer_cluster(row):
    if row["prix_moyen"] < 100:
        return "Très low-cost"
    elif row["prix_moyen"] < 200:
        return "Low-cost"
    elif row["prix_moyen"] < 400:
        return "Mid-range"
    elif row["prix_moyen"] < 800:
        return "Premium"
    else:
        return "Luxe"

profils["nom"] = profils.apply(nommer_cluster, axis=1)
df["cluster_nom"] = df["cluster"].map(profils["nom"])

print("\nRépartition par cluster et boutique :")
print(df.groupby(["cluster_nom", "shop"]).size().to_string())

# Visualisation
plt.figure(figsize=(10, 6))
palette = {
    "Très low-cost": "#27ae60",
    "Low-cost":      "#2ecc71",
    "Mid-range":     "#3498db",
    "Premium":       "#e67e22",
    "Luxe":          "#e74c3c"
}
for nom, grp in df.groupby("cluster_nom"):
    plt.scatter(grp["log_prix"], grp["score_popularite"],
                label=nom, alpha=0.5, s=15,
                color=palette.get(nom, "gray"))
plt.xlabel("Log Prix")
plt.ylabel("Score popularité")
plt.title(f"KMeans K=5 — Segmentation produits (Silhouette={sil:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig("clustering_kmeans_k5.png", dpi=150)
plt.close()
print("  ✅ clustering_kmeans_k5.png")

df.to_csv("dataset_clusters.csv", index=False, encoding="utf-8-sig")

# ============================================================
# MODULE 2 — RANDOM FOREST CORRIGÉ
# On retire score_popularite des features (fuite de données)
# ============================================================
print("\n" + "=" * 55)
print("MODULE 2 — RANDOM FOREST (corrigé)")
print("=" * 55)

# ⚠️ score_popularite EXCLU — il est calculé depuis top_k
# On garde uniquement des variables vraiment indépendantes
features_rf = ["log_prix", "promo_bin", "stock_bin",
               "shop_id", "categorie_id", "platform_id", "cluster"]

X = df[features_rf].fillna(0)
y = df["top_k"]

print(f"Répartition Top-K : {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=8           # limite la profondeur pour éviter le surapprentissage
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred,
      target_names=["Non Top-K", "Top-K"]))

# Validation croisée — plus fiable qu'un seul train/test
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1_weighted")
print(f"F1 moyen (validation croisée 5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non Top-K","Top-K"],
            yticklabels=["Non Top-K","Top-K"])
plt.title("Matrice de confusion — Random Forest corrigé")
plt.ylabel("Réel") ; plt.xlabel("Prédit")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("  ✅ confusion_matrix.png")

# Importance des features
importances = pd.Series(rf.feature_importances_,
                        index=features_rf).sort_values(ascending=True)
plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="steelblue")
plt.title("Importance des variables — Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("  ✅ feature_importance.png")

# ============================================================
# MODULE 3 — APRIORI (inchangé, résultats déjà bons)
# ============================================================
print("\n" + "=" * 55)
print("MODULE 3 — RÈGLES D'ASSOCIATION APRIORI")
print("=" * 55)

transactions = (
    df.groupby("shop")["categorie"]
    .apply(lambda x: list(x.unique()))
    .tolist()
)

te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_te = pd.DataFrame(te_array, columns=te.columns_)

frequent = apriori(df_te, min_support=0.2, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1.0)
rules = rules.sort_values("lift", ascending=False)

print(f"{len(rules)} règles trouvées")
print("\nTop 10 règles (lift le plus élevé) :")
print(rules[["antecedents","consequents","support","confidence","lift"]].head(10).to_string())

rules.to_csv("regles_association.csv", index=False, encoding="utf-8-sig")
print("  ✅ regles_association.csv")

# ============================================================
# RÉSUMÉ
# ============================================================
print("\n" + "=" * 55)
print("✅ ML PIPELINE CORRIGÉ TERMINÉ")
print("=" * 55)
print(f"  Silhouette Score KMeans K=5 : {sil:.3f}")
print(f"  F1 Random Forest (CV)       : {cv_scores.mean():.3f}")
print(f"  Règles Apriori trouvées     : {len(rules)}")