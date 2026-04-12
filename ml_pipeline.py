
# ml_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
# MODULE 1 — CLUSTERING KMEANS (K=5)
# ============================================================
print("=" * 55)
print("MODULE 1 — CLUSTERING KMEANS (K=5)")
print("=" * 55)

features_cluster = ["log_prix", "promo_bin", "stock_bin", "categorie_id", "shop_id"]
X_cluster = df[features_cluster].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df["cluster"])
print(f"Silhouette Score (K=5) : {sil:.3f}")

profils = df.groupby("cluster").agg(
    prix_moyen=("prix", "mean"),
    score_moyen=("score_popularite", "mean"),
    promo_pct=("promo_bin", "mean"),
    nb_produits=("produit", "count")
).round(2)
print("\nProfil des clusters :")
print(profils.to_string())

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
# MODULE 2 — RANDOM FOREST
# ============================================================
print("\n" + "=" * 55)
print("MODULE 2 — RANDOM FOREST")
print("=" * 55)

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
    max_depth=8
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred,
      target_names=["Non Top-K", "Top-K"]))

cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1_weighted")
print(f"F1 moyen (validation croisée 5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non Top-K", "Top-K"],
            yticklabels=["Non Top-K", "Top-K"])
plt.title("Matrice de confusion — Random Forest")
plt.ylabel("Réel")
plt.xlabel("Prédit")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("  ✅ confusion_matrix.png")

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
# MODULE 3 — RÈGLES D'ASSOCIATION APRIORI
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
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).to_string())

rules.to_csv("regles_association.csv", index=False, encoding="utf-8-sig")
print("  ✅ regles_association.csv")


# ============================================================
# MODULE 4 — PCA + DBSCAN
# ============================================================
print("\n" + "=" * 55)
print("MODULE 4 — PCA + DBSCAN")
print("=" * 55)

# On utilise plus de features que KMeans
# car DBSCAN bénéficie de plus d'information
features_db = ["log_prix", "promo_bin", "stock_bin",
               "categorie_id", "shop_id", "remise_norm", "score_popularite"]

X_db = df[features_db].fillna(0)

scaler_db = StandardScaler()
X_db_scaled = scaler_db.fit_transform(X_db)

# ── PCA ─────────────────────────────────────────────────────
print("\n── PCA ──")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_db_scaled)

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100
var_total = var1 + var2

print(f"Variance expliquée PC1 : {var1:.1f}%")
print(f"Variance expliquée PC2 : {var2:.1f}%")
print(f"Variance totale        : {var_total:.1f}%")

# Contribution des variables
print("\nContribution des variables aux composantes :")
composantes = pd.DataFrame(
    pca.components_.T,
    index=features_db,
    columns=["PC1", "PC2"]
).round(3)
print(composantes.to_string())

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

# ── DBSCAN ──────────────────────────────────────────────────
print("\n── DBSCAN ──")
print(f"\n{'eps':>6} | {'clusters':>8} | {'anomalies':>9} | {'% anomalies':>11}")
print("-" * 42)

meilleur_eps = 0.5
meilleur_score = -1

for eps in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
    db_test = DBSCAN(eps=eps, min_samples=5)
    labels_test = db_test.fit_predict(X_db_scaled)
    n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
    n_anomalies_test = (labels_test == -1).sum()
    pct = n_anomalies_test / len(labels_test) * 100
    print(f"{eps:>6} | {n_clusters_test:>8} | {n_anomalies_test:>9} | {pct:>10.1f}%")

    if 2 <= n_clusters_test <= 8 and pct < 15 and n_clusters_test > meilleur_score:
        meilleur_eps = eps
        meilleur_score = n_clusters_test

print(f"\n→ Meilleur eps choisi : {meilleur_eps}")

dbscan = DBSCAN(eps=meilleur_eps, min_samples=5)
df["dbscan_label"] = dbscan.fit_predict(X_db_scaled)

n_clusters_db = len(set(df["dbscan_label"])) - (1 if -1 in df["dbscan_label"].values else 0)
n_anomalies = (df["dbscan_label"] == -1).sum()
pct_anomalies = n_anomalies / len(df) * 100

print(f"\nRésultats DBSCAN :")
print(f"  Clusters détectés : {n_clusters_db}")
print(f"  Anomalies         : {n_anomalies} ({pct_anomalies:.1f}%)")

# Profil des clusters
print("\nProfil des clusters DBSCAN :")
profil_db = df.groupby("dbscan_label").agg(
    nb_produits=("produit", "count"),
    prix_moyen=("prix", "mean"),
    score_moyen=("score_popularite", "mean"),
    promo_pct=("promo_bin", "mean")
).round(2)
profil_db.index = ["Anomalie" if i == -1 else f"Cluster {i}"
                   for i in profil_db.index]
print(profil_db.to_string())

# Silhouette Score DBSCAN
df_sans_bruit = df[df["dbscan_label"] != -1]
if len(df_sans_bruit["dbscan_label"].unique()) > 1:
    sil_db = silhouette_score(
        X_db_scaled[df["dbscan_label"] != -1],
        df_sans_bruit["dbscan_label"]
    )
    print(f"\nSilhouette Score DBSCAN : {sil_db:.3f}")
else:
    sil_db = 0
    print("\nPas assez de clusters pour le silhouette score")

# Top anomalies
anomalies = df[df["dbscan_label"] == -1].copy()
print(f"\nTop 10 anomalies détectées :")
print(anomalies.nlargest(10, "prix")[
    ["shop", "categorie", "produit", "prix", "remise_norm", "score_popularite"]
].to_string())

# ── Visualisations PCA + DBSCAN ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Graphique 1 : PCA coloré par KMeans
ax1 = axes[0]
for nom, grp in df.groupby("cluster_nom"):
    ax1.scatter(grp["pca1"], grp["pca2"],
                label=nom, alpha=0.4, s=8,
                color=palette.get(nom, "gray"))
ax1.legend(markerscale=3, fontsize=8)
ax1.set_xlabel(f"PC1 ({var1:.1f}%)")
ax1.set_ylabel(f"PC2 ({var2:.1f}%)")
ax1.set_title("PCA — Clusters KMeans")

# Graphique 2 : PCA coloré par DBSCAN
ax2 = axes[1]
couleurs_db = {-1: "#e74c3c"}
palette_db = ["#3498db", "#27ae60", "#e67e22", "#9b59b6", "#1abc9c", "#f39c12"]
for i, label in enumerate([l for l in sorted(df["dbscan_label"].unique()) if l != -1]):
    couleurs_db[label] = palette_db[i % len(palette_db)]

for label, grp in df.groupby("dbscan_label"):
    nom = "Anomalie" if label == -1 else f"Cluster {label}"
    ax2.scatter(grp["pca1"], grp["pca2"],
                label=f"{nom} ({len(grp)})",
                color=couleurs_db[label],
                alpha=0.8 if label == -1 else 0.4,
                s=20 if label == -1 else 8)
ax2.legend(markerscale=2, fontsize=8)
ax2.set_xlabel(f"PC1 ({var1:.1f}%)")
ax2.set_ylabel(f"PC2 ({var2:.1f}%)")
ax2.set_title(f"PCA — DBSCAN\n{n_anomalies} anomalies en rouge")

# Graphique 3 : Variance expliquée cumulée
ax3 = axes[2]
pca_full = PCA(random_state=42)
pca_full.fit(X_db_scaled)
var_cumul = np.cumsum(pca_full.explained_variance_ratio_) * 100
ax3.plot(range(1, len(var_cumul) + 1), var_cumul, "bo-", linewidth=2, markersize=6)
ax3.axhline(y=80, color="red", linestyle="--", alpha=0.7, label="Seuil 80%")
ax3.axhline(y=95, color="orange", linestyle="--", alpha=0.7, label="Seuil 95%")
ax3.axvline(x=2, color="green", linestyle="--", alpha=0.7, label="2 composantes")
ax3.fill_between(range(1, len(var_cumul) + 1), var_cumul, alpha=0.1, color="blue")
ax3.set_xlabel("Nombre de composantes")
ax3.set_ylabel("Variance expliquée (%)")
ax3.set_title("PCA — Variance expliquée cumulée")
ax3.legend(fontsize=8)
ax3.set_xticks(range(1, len(var_cumul) + 1))
ax3.grid(True, alpha=0.3)

plt.suptitle("PCA + DBSCAN — Smart eCommerce", fontsize=13)
plt.tight_layout()
plt.savefig("dbscan_pca_visualisation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ dbscan_pca_visualisation.png")

# Graphique anomalies par shop
anomalies_shop = df.groupby("shop").apply(
    lambda x: (x["dbscan_label"] == -1).sum()
).reset_index()
anomalies_shop.columns = ["shop", "nb_anomalies"]

plt.figure(figsize=(8, 4))
plt.bar(anomalies_shop["shop"], anomalies_shop["nb_anomalies"],
        color=["#3498db", "#27ae60", "#e67e22", "#9b59b6", "#1abc9c"])
plt.title("Anomalies DBSCAN par boutique")
plt.ylabel("Nombre d'anomalies")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("dbscan_anomalies.png", dpi=150)
plt.close()
print("  ✅ dbscan_anomalies.png")

# Sauvegarde anomalies
anomalies[["shop", "categorie", "produit", "prix",
           "remise_norm", "score_popularite"]].to_csv(
    "anomalies_dbscan.csv", index=False, encoding="utf-8-sig"
)
print("  ✅ anomalies_dbscan.csv")

# Sauvegarde dataset final
df.to_csv("dataset_features.csv", index=False, encoding="utf-8-sig")
print("  ✅ dataset_features.csv mis à jour")


# ============================================================
# RÉSUMÉ FINAL
# ============================================================
print("\n" + "=" * 55)
print("✅ ML PIPELINE COMPLET TERMINÉ")
print("=" * 55)
print(f"  Silhouette Score KMeans K=5  : {sil:.3f}")
print(f"  F1 Random Forest (CV)        : {cv_scores.mean():.3f}")
print(f"  Règles Apriori trouvées      : {len(rules)}")
print(f"  Variance PCA (2 composantes) : {var_total:.1f}%")
print(f"  Clusters DBSCAN              : {n_clusters_db}")
print(f"  Anomalies DBSCAN             : {n_anomalies} ({pct_anomalies:.1f}%)")
if sil_db > 0:
    print(f"  Silhouette Score DBSCAN      : {sil_db:.3f}")
print(f"\nFichiers générés :")
print(f"  → clustering_kmeans_k5.png")
print(f"  → confusion_matrix.png")
print(f"  → feature_importance.png")
print(f"  → regles_association.csv")
print(f"  → dbscan_pca_visualisation.png")
print(f"  → dbscan_anomalies.png")
print(f"  → anomalies_dbscan.csv")
print(f"  → dataset_clusters.csv")
print(f"  → dataset_features.csv")