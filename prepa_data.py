import pandas as pd
import numpy as np
import re

# ============================================================
# ÉTAPE 1 — FUSION INTELLIGENTE
# ============================================================

def charger_kiabi(fichier):
    # ⚠️ Kiabi a des virgules dans les noms → on force engine python
    # + on ignore les lignes mal formées
    df = pd.read_csv(
        fichier,
        encoding="utf-8-sig",
        quotechar='"',
        engine="python",        # plus robuste que le parser C
        on_bad_lines="skip"     # ignore les lignes corrompues
    )
    df["sous_categorie"] = df["categorie"]
    df["ancien_prix"]    = 0
    df["remise_pct"]     = "N/A"
    df["en_promo"]       = "Non"
    df["marque"]         = "N/A"
    df["url_produit"]    = "N/A"
    df["platform"]       = "Shopify"
    return df

def charger_beautymarket(fichier):
    df = pd.read_csv(fichier, encoding="utf-8-sig", quotechar='"', engine="python", on_bad_lines="skip")
    df["ancien_prix"] = 0
    df["remise_pct"]  = "N/A"
    df["en_promo"]    = "Non"
    df["marque"]      = "N/A"
    df["platform"]    = "Shopify"
    return df

def charger_justyol(fichier):
    df = pd.read_csv(fichier, encoding="utf-8-sig", quotechar='"', engine="python", on_bad_lines="skip")
    df["platform"] = "Shopify"
    return df

def charger_woocommerce(fichier):
    df = pd.read_csv(fichier, encoding="utf-8-sig", quotechar='"', engine="python", on_bad_lines="skip")
    df["sous_categorie"] = df["categorie"]
    df["marque"]         = "N/A"
    df["url_produit"]    = "N/A"
    df["platform"]       = "WooCommerce"
    return df

# Colonnes communes finales
COLONNES = [
    "shop", "platform", "categorie", "sous_categorie",
    "marque", "produit", "prix", "ancien_prix",
    "remise_pct", "stock", "en_promo", "rating", "url_produit"
]

print("=" * 50)
print("ÉTAPE 1 — CHARGEMENT DES FICHIERS")
print("=" * 50)
dfs = []

for nom, fn in [
    ("dataset_kiabi_final.csv",   charger_kiabi),
    ("dataset_beautymarket.csv",  charger_beautymarket),
    ("dataset_justyol.csv",       charger_justyol),
    ("dataset_woocommerce.csv",   charger_woocommerce),
]:
    try:
        df = fn(nom)
        # Ajouter colonnes manquantes
        for col in COLONNES:
            if col not in df.columns:
                df[col] = "N/A"
        dfs.append(df[COLONNES])
        print(f"  ✅ {nom} — {len(df)} lignes")
    except Exception as e:
        print(f"  ❌ {nom} : {e}")

df_global = pd.concat(dfs, ignore_index=True)
df_global.to_csv("dataset_global.csv", index=False, encoding="utf-8-sig")
print(f"\nTotal brut : {len(df_global)} produits")
print(df_global.groupby("shop").size().to_string())


# ============================================================
# ÉTAPE 2 — NETTOYAGE
# ============================================================
print("\n" + "=" * 50)
print("ÉTAPE 2 — NETTOYAGE")
print("=" * 50)

df = pd.read_csv("dataset_global.csv", encoding="utf-8-sig", quotechar='"', engine="python")

# Supprimer lignes vides
df = df.dropna(subset=["produit", "prix"])
df = df[df["produit"].astype(str).str.strip() != ""]
df = df[df["prix"].astype(str).str.strip() != ""]

# Corriger les prix
def corriger_prix(val):
    s = str(val).strip()
    s = s.replace(" ", "").replace("MAD","").replace("DH","").replace("dh","")
    # "1.199" → milliers → "1199"
    if re.match(r"^\d+\.\d{3}$", s):
        s = s.replace(".", "")
    return s

df["prix"] = df["prix"].apply(corriger_prix)
df["prix"] = pd.to_numeric(df["prix"], errors="coerce")
df = df.dropna(subset=["prix"])
df = df[df["prix"] > 0]

# Ancien prix
df["ancien_prix"] = df["ancien_prix"].apply(
    lambda x: 0 if str(x).strip() in ["N/A","nan",""] else x
)
df["ancien_prix"] = pd.to_numeric(df["ancien_prix"], errors="coerce").fillna(0)

# Encodages binaires
df["stock_bin"] = df["stock"].apply(lambda x: 1 if str(x).strip().lower() == "en stock" else 0)
df["promo_bin"] = df["en_promo"].apply(lambda x: 1 if str(x).strip().lower() == "oui" else 0)
df["rating"]    = pd.to_numeric(df["rating"], errors="coerce").fillna(0)

# Doublons
df = df.drop_duplicates(subset=["shop", "produit", "prix"])
df = df.reset_index(drop=True)

print(f"Après nettoyage : {len(df)} produits")
print(df.groupby("shop").size().to_string())
df.to_csv("dataset_clean.csv", index=False, encoding="utf-8-sig")


# ============================================================
# VÉRIFICATION — voir ce qui ne va pas
# ============================================================

df = pd.read_csv("dataset_clean.csv", engine="python")

# 1. Afficher les noms de shops bizarres (pas dans la liste attendue)
shops_attendus = ["Kiabi", "BeautyMarket", "Justyol", "Lasolda", "Lhmiza"]
shops_inconnus = df[~df["shop"].isin(shops_attendus)]
print(f"Lignes avec shop invalide : {len(shops_inconnus)}")
print(shops_inconnus[["shop","produit","prix"]].to_string())

# 2. Supprimer ces lignes corrompues
df = df[df["shop"].isin(shops_attendus)]
df = df.reset_index(drop=True)

print(f"\nAprès suppression lignes corrompues : {len(df)} produits")
print(df.groupby("shop").size().to_string())

# Sauvegarder le clean corrigé
df.to_csv("dataset_clean.csv", index=False, encoding="utf-8-sig")
print("\n✅ dataset_clean.csv corrigé")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "="*50)
print("ÉTAPE 3 — FEATURE ENGINEERING")
print("="*50)

df = pd.read_csv("dataset_clean.csv", engine="python")

# Log prix
df["log_prix"] = np.log1p(df["prix"])

# Segment prix
df["segment_prix"] = pd.cut(
    df["prix"],
    bins=[0, 100, 500, 99999],
    labels=["low-cost", "mid-range", "premium"]
)

# Score popularité
df["score_popularite"] = (
    df["rating"].clip(0, 5) * 0.6 +
    df["stock_bin"]          * 0.2 +
    df["promo_bin"]          * 0.2
).round(3)

# Label Top-K — top 20% par catégorie
df["top_k"] = df.groupby("categorie")["score_popularite"].transform(
    lambda x: (x >= x.quantile(0.80)).astype(int)
)

# Encodages numériques
df["shop_id"]      = pd.Categorical(df["shop"]).codes
df["categorie_id"] = pd.Categorical(df["categorie"]).codes
df["platform_id"]  = pd.Categorical(df["platform"]).codes

df.to_csv("dataset_features.csv", index=False, encoding="utf-8-sig")

print(f"✅ dataset_features.csv — {len(df)} produits")
print("\nTop-K par boutique :")
print(df.groupby("shop")["top_k"].sum().astype(int).to_string())
print("\nSegments de prix par boutique :")
print(df.groupby(["shop","segment_prix"]).size().to_string())
print("\nAperçu :")
print(df[["shop","produit","prix","segment_prix","score_popularite","top_k"]].head(10).to_string())