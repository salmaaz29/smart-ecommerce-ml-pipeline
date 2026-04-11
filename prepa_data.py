import pandas as pd
import numpy as np

# ============================================================
# FEATURE ENGINEERING v3 — Score Top-K universel
# Corrige : NaN pour Kiabi/BeautyMarket, prix aberrants
# ============================================================

df = pd.read_csv("dataset_clean.csv", engine="python")
print(f"Dataset chargé : {len(df)} produits")


# ============================================================
# ÉTAPE 1 — Supprimer les prix aberrants
# Un produit à 1.7 DH ou 1 DH est une erreur de parsing
# On garde uniquement les prix entre 5 DH et 50 000 DH
# ============================================================

avant = len(df)
df = df[(df["prix"] >= 5) & (df["prix"] <= 50000)]
print(f"Prix aberrants supprimés : {avant - len(df)} lignes")
print(f"Produits restants : {len(df)}")


# ============================================================
# ÉTAPE 2 — Forcer rating = 0 partout (aucun rating fiable)
# Kiabi = fictif, les autres = N/A
# On construit le score SANS rating pour tout le monde
# ============================================================

df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)

# Kiabi avait 4.2 fictif → on remet à 0
df.loc[df["shop"] == "Kiabi", "rating"] = 0

print("\n── Rating après correction ──")
print(df.groupby("shop")["rating"].apply(
    lambda x: f"{(x > 0).sum()} valides / {len(x)} total"
).to_string())


# ============================================================
# ÉTAPE 3 — Normalisation du prix PAR CATÉGORIE
# Prix bas dans sa catégorie = signal positif
# ============================================================

df["prix_score"] = df.groupby("categorie")["prix"].rank(
    ascending=True, method="min", pct=True
)
df["prix_score"] = (1 - df["prix_score"]).round(4)  # inversion : bas = mieux


# ============================================================
# ÉTAPE 4 — Normalisation de la remise
# ============================================================

def parse_remise(val):
    try:
        if pd.isna(val):        # ← ajouter cette ligne
            return 0.0
        v = float(str(val).replace("%", "").strip())
        return min(v, 100) / 100
    except:
        return 0.0

df["remise_norm"] = df["remise_pct"].apply(parse_remise)


# ============================================================
# ÉTAPE 5 — Score universel (même formule pour tous les shops)
#
# On n'utilise PAS le rating car aucun shop n'a de rating fiable
#
# score = stock_bin  × 0.30   → produit disponible ?
#       + promo_bin  × 0.25   → en promotion ?
#       + prix_score × 0.25   → prix compétitif dans sa catégorie ?
#       + remise_norm× 0.20   → remise réelle détectée ?
#
# Résultat : valeur entre 0 et 1 pour TOUS les produits
# ============================================================

df["score_popularite"] = (
    df["stock_bin"]   * 0.30 +
    df["promo_bin"]   * 0.25 +
    df["prix_score"]  * 0.25 +
    df["remise_norm"] * 0.20
).round(4)

print("\n── Statistiques score_popularite par shop ──")
print(df.groupby("shop")["score_popularite"].describe().round(3).to_string())


# ============================================================
# ÉTAPE 6 — Label Top-K (top 20% par catégorie)
# ============================================================

df["top_k"] = df.groupby("categorie")["score_popularite"].transform(
    lambda x: (x >= x.quantile(0.80)).astype(int)
)

print("\n── Top-K par boutique ──")
print(df.groupby("shop")["top_k"].agg(["sum", "count"])
      .rename(columns={"sum": "top_k_count", "count": "total"})
      .assign(pct=lambda x: (x["top_k_count"] / x["total"] * 100).round(1))
      .to_string())


# ============================================================
# ÉTAPE 7 — Autres features
# ============================================================

df["log_prix"]     = np.log1p(df["prix"])

df["segment_prix"] = pd.cut(
    df["prix"],
    bins=[0, 100, 500, 99999],
    labels=["low-cost", "mid-range", "premium"]
)

df["shop_id"]      = pd.Categorical(df["shop"]).codes
df["categorie_id"] = pd.Categorical(df["categorie"]).codes
df["platform_id"]  = pd.Categorical(df["platform"]).codes


# ============================================================
# ÉTAPE 8 — Vérification finale
# ============================================================

nan_score = df["score_popularite"].isna().sum()
print(f"\nProduits avec score NaN : {nan_score}  ← doit être 0 ✅")

print(f"\nTotal produits scorés : {len(df)}")
print(f"Dont Top-K            : {df['top_k'].sum()}")

# Aperçu Top-K
print("\n── Top 15 produits (score le plus élevé) ──")
cols = ["shop", "categorie", "produit", "prix",
        "stock_bin", "promo_bin", "remise_norm", "score_popularite"]
top = (df[df["top_k"] == 1]
       .sort_values("score_popularite", ascending=False)
       .head(15)[cols])
print(top.to_string())


# ============================================================
# SAUVEGARDE
# ============================================================

df.to_csv("dataset_features.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ dataset_features.csv sauvegardé — {len(df)} produits")