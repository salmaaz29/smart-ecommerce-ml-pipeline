# beauty_market_scrapper 


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

COLLECTIONS = {
    "Maquillage": [
        "https://beautymarket.ma/collections/fond-de-teint",
        "https://beautymarket.ma/collections/anti-cernes-correcteur",
        "https://beautymarket.ma/collections/rougir",
        "https://beautymarket.ma/collections/palette",
        "https://beautymarket.ma/collections/highlight",
        "https://beautymarket.ma/collections/palette-yeux",
        "https://beautymarket.ma/collections/mascara",
        "https://beautymarket.ma/collections/eyeliner",
        "https://beautymarket.ma/collections/crayon-yeux",
        "https://beautymarket.ma/collections/rouge-a-levres",
        "https://beautymarket.ma/collections/gloss",
        "https://beautymarket.ma/collections/crayon-a-levres",
        "https://beautymarket.ma/collections/levres-plump",
        "https://beautymarket.ma/collections/rouge-a-levres-liquide",
        "https://beautymarket.ma/collections/rouge-a-levres-mat",
        "https://beautymarket.ma/collections/ongles",
        "https://beautymarket.ma/collections/outils-et-accessoires-de-beaute",
    ]
}

def get_sous_categorie(url):
    return url.rstrip("/").split("/")[-1].replace("-", " ").title()

def nettoyer_prix(texte):
    """Extrait le premier nombre valide d'un texte de prix."""
    texte = texte.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    match = re.search(r"\d+[.,]?\d*", texte)
    return match.group().replace(",", ".") if match else ""

def scrape_collection(driver, url, cat_parente, sous_cat):
    produits = []
    page = 1

    while True:
        page_url = f"{url}?page={page}"
        print(f"    [{cat_parente} > {sous_cat}] Page {page}")
        driver.get(page_url)

        try:
            # ✅ Sélecteur exact vu dans l'inspecteur
            WebDriverWait(driver, 12).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.product-item"))
            )
        except:
            print(f"    ⚠️  Timeout ou fin des pages.")
            break

        # ✅ Cards : div.product-item (confirmé dans le HTML)
        cards = driver.find_elements(By.CSS_SELECTOR, "div.product-item")

        if not cards:
            break

        nb_avant = len(produits)

        for card in cards:
            try:
                # ✅ NOM : h3.product-item__product-title (confirmé)
                nom = ""
                try:
                    nom = card.find_element(
                        By.CSS_SELECTOR, "h3.product-item__product-title"
                    ).text.strip()
                except:
                    # fallback si classe légèrement différente
                    try:
                        nom = card.find_element(By.CSS_SELECTOR, "h3").text.strip()
                    except:
                        pass

                # ✅ PRIX : dans product-item__text_group_secondary
                # Sur ce thème Shopify (Impulse/Prestige) le prix est dans .price
                prix_num = ""
                try:
                    prix_brut = card.find_element(
                        By.CSS_SELECTOR,
                        ".product-item__text_group_secondary .price, "
                        ".product-item__price, "
                        ".price__current, "
                        "span.price"
                    ).text.strip()
                    prix_num = nettoyer_prix(prix_brut)
                except:
                    pass

                # ✅ STOCK : data-show-inventory="false" par défaut
                # On regarde si un badge "Épuisé" est présent
                stock = "En stock"
                try:
                    badge = card.find_element(
                        By.CSS_SELECTOR, ".product-item__sold-out-badge, .badge--sold-out"
                    ).text.strip().lower()
                    if badge:
                        stock = "Épuisé"
                except:
                    pass

                # ✅ RATING : data-ratings-visible="false" sur ce site → N/A
                rating = "N/A"
                try:
                    r = card.find_element(
                        By.CSS_SELECTOR, ".jdgm-prev-badge, [class*='rating']"
                    )
                    rating = r.get_attribute("data-score") or r.text.strip() or "N/A"
                except:
                    pass

                # ✅ URL PRODUIT (bonus utile pour enrichissement LLM)
                url_produit = ""
                try:
                    href = card.find_element(By.CSS_SELECTOR, "a.product-item__image-wrapper, a[href*='/products/']").get_attribute("href")
                    url_produit = href if href else ""
                except:
                    pass

                if nom and prix_num:
                    produits.append({
                        "shop": "BeautyMarket",
                        "categorie": cat_parente,
                        "sous_categorie": sous_cat,
                        "produit": nom,
                        "prix": prix_num,
                        "stock": stock,
                        "rating": rating,
                        "url_produit": url_produit
                    })

            except:
                continue

        if len(produits) == nb_avant:
            print(f"    Page vide — fin de collection.")
            break

        print(f"    ✅ +{len(produits) - nb_avant} produits (total : {len(produits)})")
        page += 1
        time.sleep(2.5)

    return produits


def scrape_beautymarket_full():
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

            # Anti-doublons inter-collections
            for p in produits_collection:
                cle = (p["produit"].lower(), p["prix"])
                if cle not in produits_vus:
                    produits_vus.add(cle)
                    tous_les_produits.append(p)

            print(f"  → {len(produits_collection)} scrapés | {len(tous_les_produits)} uniques total")
            time.sleep(1.5)

    driver.quit()
    return tous_les_produits


import os


if __name__ == "__main__":
    resultats = scrape_beautymarket_full()
    df = pd.DataFrame(resultats)
   # Vérifier si le fichier existe déjà
    file_path = "dataset_beautymarket.csv"
    if not os.path.isfile(file_path):
        # Première fois → créer avec en-tête
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    else:
        # Ajouter sans écraser, sans en-tête
        df.to_csv(file_path, mode="a", index=False, header=False, encoding="utf-8-sig")

    print(f"\n✅ {len(df)} nouveaux produits ajoutés → {file_path}")
    print("\n── Répartition par catégorie ──")
    print(df.groupby(["categorie", "sous_categorie"]).size().to_string())
