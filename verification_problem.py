import pandas as pd
import numpy as np

df = pd.read_csv("dataset_clean.csv", engine="python")
df = df[(df["prix"] >= 5) & (df["prix"] <= 50000)]
df.loc[df["shop"] == "Kiabi", "rating"] = 0

# Calcul pas à pas
df["prix_score"] = df.groupby("categorie")["prix"].rank(
    ascending=True, method="min", pct=True
)
df["prix_score"] = (1 - df["prix_score"]).round(4)

def parse_remise(val):
    try:
        v = float(str(val).replace("%", "").strip())
        return min(v, 100) / 100
    except:
        return 0.0

df["remise_norm"] = df["remise_pct"].apply(parse_remise)

# Vérifier chaque composante pour Kiabi
kiabi = df[df["shop"] == "Kiabi"]
print("── Kiabi — vérification composantes ──")
print(f"stock_bin  NaN : {kiabi['stock_bin'].isna().sum()}")
print(f"promo_bin  NaN : {kiabi['promo_bin'].isna().sum()}")
print(f"prix_score NaN : {kiabi['prix_score'].isna().sum()}")
print(f"remise_norm NaN: {kiabi['remise_norm'].isna().sum()}")

print("\n── Types ──")
print(f"stock_bin  dtype: {df['stock_bin'].dtype}")
print(f"promo_bin  dtype: {df['promo_bin'].dtype}")
print(f"prix_score dtype: {df['prix_score'].dtype}")
print(f"remise_norm dtype: {df['remise_norm'].dtype}")

# Calcul du score
df["score_popularite"] = (
    df["stock_bin"]   * 0.30 +
    df["promo_bin"]   * 0.25 +
    df["prix_score"]  * 0.25 +
    df["remise_norm"] * 0.20
).round(4)

print("\n── Score Kiabi après calcul ──")
print(f"NaN : {df[df['shop']=='Kiabi']['score_popularite'].isna().sum()}")
print(df[df["shop"]=="Kiabi"]["score_popularite"].describe())

print("\n── Score BeautyMarket après calcul ──")
print(f"NaN : {df[df['shop']=='BeautyMarket']['score_popularite'].isna().sum()}")
print(df[df["shop"]=="BeautyMarket"]["score_popularite"].describe())