# justyol_scrapper 


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import re
import time

def setup_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Désactiver pour déboguer
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ============================================================
# COLLECTIONS JUSTYOL — confirmées depuis le site
# ============================================================
COLLECTIONS = {
    "Ventes Flash":   ["https://justyol.com/collections/all-best-selling"],
    "Femme":          ["https://justyol.com/collections/women"],
    "Homme":          ["https://justyol.com/collections/men"],
    "Enfants":        ["https://justyol.com/collections/kids-babies-all"],
    "Maison":         ["https://justyol.com/collections/home-all"],
    "Beauté & Santé": ["https://justyol.com/collections/beaute-soin"],
}

def get_sous_categorie(url):
    return url.rstrip("/").split("/")[-1].replace("-", " ").title()

def nettoyer_prix(texte):
    """Extrait le premier nombre valide — gère '112.00 Dhs', '165.00 Dhs'"""
    texte = texte.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    match = re.search(r"\d+[.,]?\d*", texte)
    return match.group().replace(",", ".") if match else ""

def scrape_collection(driver, url, cat_parente, sous_cat):
    produits = []
    page = 1
    MAX_PAGES_FLASH = 10

    while True:
        if cat_parente == "Ventes Flash" and page > MAX_PAGES_FLASH:
            print(f"    Limite atteinte ({MAX_PAGES_FLASH} pages) pour Ventes Flash.")
            break

        page_url = f"{url}?page={page}"
        print(f"    [{cat_parente}] Page {page} → {page_url}")
        driver.get(page_url)

        try:
            # ✅ Sélecteur confirmé : div.product-card
            WebDriverWait(driver, 12).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.product-card"))
            )
        except:
            print(f"    ⚠️  Timeout ou fin des pages pour {cat_parente}.")
            break

        # ✅ Cards confirmées : div.f-column.product-column > div.product-card
        cards = driver.find_elements(By.CSS_SELECTOR, "div.product-card")

        if not cards:
            print(f"    Aucune card trouvée — fin.")
            break

        nb_avant = len(produits)

        for card in cards:
            try:
                # ── NOM ──────────────────────────────────────────────────
                # ✅ Confirmé : h3.product-card__title > a.reversed-link
                nom = ""
                try:
                    nom = card.find_element(
                        By.CSS_SELECTOR, "h3.product-card__title a"
                    ).text.strip()
                except:
                    try:
                        nom = card.find_element(By.CSS_SELECTOR, "h3").text.strip()
                    except:
                        pass

                # ── MARQUE / VENDEUR ──────────────────────────────────────
                # ✅ Confirmé : p.product-card__vendor > a
                marque = ""
                try:
                    marque = card.find_element(
                        By.CSS_SELECTOR, "p.product-card__vendor a"
                    ).text.strip()
                except:
                    pass

                # ── PRIX ──────────────────────────────────────────────────
                # ✅ Confirmé dans l'HTML :
                #   - Prix régulier : span.f-price-item.f-price-item--regular → "112.00 Dhs"
                #   - Prix soldé    : span.f-price-item.f-price-item--sale    → "112.00 Dhs"
                #   - Ancien prix   : span barré dans f-price-item--regular après f-price__sale
                prix_actuel = ""
                ancien_prix = ""

                try:
                    # Prix de vente (soldé ou régulier affiché)
                    prix_el = card.find_element(
                        By.CSS_SELECTOR,
                        "div.f-price__sale span.f-price-item--sale, "
                        "div.f-price__regular span.f-price-item--regular"
                    )
                    prix_actuel = nettoyer_prix(prix_el.text)
                except:
                    pass

                try:
                    # Ancien prix barré (s > span dans f-price-item--regular)
                    ancien_el = card.find_element(
                        By.CSS_SELECTOR, "div.f-price__sale span.f-price-item--regular s"
                    )
                    ancien_prix = nettoyer_prix(ancien_el.text)
                except:
                    pass

                # Calcul remise si les deux prix sont disponibles
                remise = ""
                try:
                    if prix_actuel and ancien_prix:
                        p_act = float(prix_actuel)
                        p_anc = float(ancien_prix)
                        if p_anc > 0 and p_anc > p_act:
                            remise = str(round((1 - p_act / p_anc) * 100))
                except:
                    pass

                # ── STOCK ─────────────────────────────────────────────────
                stock = "En stock"
                try:
                    badge = card.find_element(
                        By.CSS_SELECTOR,
                        ".product-card__badge-soldout, [class*='sold-out'], [class*='unavailable']"
                    ).text.strip().lower()
                    if badge:
                        stock = "Épuisé"
                except:
                    pass

                # ── RATING ────────────────────────────────────────────────
                rating = "N/A"
                try:
                    r = card.find_element(
                        By.CSS_SELECTOR, "[class*='rating'], [class*='star'], .jdgm-prev-badge"
                    )
                    rating = r.get_attribute("data-score") or r.text.strip() or "N/A"
                except:
                    pass

                # ── URL PRODUIT ───────────────────────────────────────────
                url_produit = ""
                try:
                    url_produit = card.find_element(
                        By.CSS_SELECTOR, "h3.product-card__title a"
                    ).get_attribute("href") or ""
                except:
                    pass

                if nom and prix_actuel:
                    produits.append({
                        "shop":          "Justyol",
                        "categorie":     cat_parente,
                        "sous_categorie": sous_cat,
                        "produit":       nom,
                        "marque":        marque,
                        "prix":          prix_actuel,
                        "ancien_prix":   ancien_prix if ancien_prix else "N/A",
                        "remise_pct":    remise if remise else "N/A",
                        "stock":         stock,
                        "rating":        rating,
                        "url_produit":   url_produit
                    })

            except:
                continue

        if len(produits) == nb_avant:
            print(f"    Page {page} vide — fin de collection.")
            break

        print(f"    ✅ +{len(produits) - nb_avant} produits (total collection : {len(produits)})")
        page += 1
        time.sleep(2.5)

    return produits


def scrape_justyol_full():
    driver = setup_driver()
    tous_les_produits = []
    produits_vus = set()

    for cat_parente, urls in COLLECTIONS.items():
        print(f"\n{'='*55}")
        print(f"  CATÉGORIE : {cat_parente}")
        print(f"{'='*55}")

        for url in urls:
            sous_cat = get_sous_categorie(url)
            produits_collection = scrape_collection(driver, url, cat_parente, sous_cat)

            for p in produits_collection:
                cle = (p["produit"].lower(), p["prix"])
                if cle not in produits_vus:
                    produits_vus.add(cle)
                    tous_les_produits.append(p)

            print(f"  → {len(produits_collection)} scrapés | {len(tous_les_produits)} uniques total")
            time.sleep(1.5)

    driver.quit()
    return tous_les_produits


if __name__ == "__main__":
    resultats = scrape_justyol_full()
    df = pd.DataFrame(resultats)
    df.to_csv("dataset_justyol.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ {len(df)} produits uniques → dataset_justyol.csv")
    print("\n── Répartition par catégorie ──")
    print(df.groupby(["categorie"]).size().to_string())