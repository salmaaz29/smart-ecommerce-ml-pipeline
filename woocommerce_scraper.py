# site woocommerce_scrapper 


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
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ============================================================
# URLS CONFIRMÉES ET CORRIGÉES
# ============================================================
SITES = {
    "Lasolda": {
        "scraper": "lasolda",
        "categories": {
            "Maison":          [
                "https://lasolda.ma/product-category/maison/electromenager/",
                "https://lasolda.ma/product-category/maison/accessoires-d-exterieur/",
                "https://lasolda.ma/product-category/maison/rangements/",
            ],
            "Beauté & Sport":  [
                "https://lasolda.ma/product-category/beaute-bien-etre/",
            ],
            "Ventes flash":         [
                "https://lasolda.ma/product-category/ventes-flash/",
            ],
            "Automobile et bricolage":         [
                "https://lasolda.ma/product-category/automobile-bricolage/",
            ],
            "Bagages":         [
                "https://lasolda.ma/product-category/bagages-cartables/",
            ],
            "Les plus demandés":       ["https://lasolda.ma/product-category/plus-demandes/"]
        }
    },

    "Lhmiza": {
        "scraper": "lhmiza",
        "categories": {
            "Beauté":                  ["https://lhmiza.ma/fr/produits-beaute-maroc/"],
            "Maquillage":              ["https://lhmiza.ma/fr/produits-beaute-maroc/maquillage/"],
            "Parfums":                 ["https://lhmiza.ma/fr/parfums-maroc/"],
            "Montres & Bijoux":        ["https://lhmiza.ma/fr/montres-et-bijoux-maroc/"],
            "Téléphones & Accessoires":["https://lhmiza.ma/fr/produits-high-tech/telephones-et-accessoires/"]        }
    }
}

# ============================================================
# UTILITAIRES
# ============================================================
def nettoyer_prix(texte):
    texte = (texte
             .replace("\u202f", "").replace("\xa0", "")
             .replace("MAD", "").replace("DH", "").replace("dh", "")
             .replace("د.م.", "").replace(" ", "").replace(",", "."))
    match = re.search(r"\d+\.?\d*", texte)
    return match.group() if match else ""

# ============================================================
# SCRAPER LASOLDA
# Confirmé screenshot 1 :
#   card  → div.first.grid-sizer  (ou div[class*="grid-sizer"])
#   nom   → h2.product-title > a
#   prix  → span.price > span.woocommerce-Price-amount.amount
#   stock → classe "instock" ou "outofstock" sur la card
#   promo → classe "sale" sur la card
# ============================================================
def scrape_cards_lasolda(driver, categorie):
    produits = []

    try:
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.ajax-content, div.products-loop")
            )
        )
    except:
        print("    ⚠️  Timeout Lasolda")
        return produits

    # ✅ Cards : div avec classe "grid-sizer" (confirmé screenshot)
    cards = driver.find_elements(By.CSS_SELECTOR, "div[class*='grid-sizer']")

    if not cards:
        # Fallback : tous les divs produits dans ajax-content
        cards = driver.find_elements(By.CSS_SELECTOR, "div.ajax-content div.content-product")

    for card in cards:
        try:
            # ✅ NOM : h2.product-title > a  (confirmé)
            nom = ""
            try:
                nom = card.find_element(By.CSS_SELECTOR, "h2.product-title a").text.strip()
            except:
                try:
                    nom = card.find_element(By.CSS_SELECTOR, "h2, h3").text.strip()
                except:
                    pass

            # ✅ PRIX : span.price > span.woocommerce-Price-amount.amount (confirmé)
            prix_actuel = ""
            ancien_prix = ""
            try:
                # Prix soldé dans <ins>
                ins = card.find_element(
                    By.CSS_SELECTOR, "span.price ins span.woocommerce-Price-amount"
                )
                prix_actuel = nettoyer_prix(ins.text)
            except:
                pass

            if not prix_actuel:
                try:
                    prix_el = card.find_element(
                        By.CSS_SELECTOR,
                        "span.price span.woocommerce-Price-amount.amount"
                    )
                    prix_actuel = nettoyer_prix(prix_el.text)
                except:
                    pass

            try:
                del_el = card.find_element(
                    By.CSS_SELECTOR, "span.price del span.woocommerce-Price-amount"
                )
                ancien_prix = nettoyer_prix(del_el.text)
            except:
                pass

            # Calcul remise
            remise = "N/A"
            try:
                if prix_actuel and ancien_prix:
                    p_act = float(prix_actuel)
                    p_anc = float(ancien_prix)
                    if p_anc > p_act > 0:
                        remise = str(round((1 - p_act / p_anc) * 100))
            except:
                pass

            # ✅ STOCK : classes sur la card (instock / outofstock)
            stock = "En stock"
            try:
                classes = card.get_attribute("class") or ""
                if "outofstock" in classes:
                    stock = "Épuisé"
            except:
                pass

            # ✅ PROMO : classe "sale" sur la card
            en_promo = "Non"
            try:
                classes = card.get_attribute("class") or ""
                if " sale " in f" {classes} ":
                    en_promo = "Oui"
                else:
                    card.find_element(By.CSS_SELECTOR, ".onsale, span.onsale")
                    en_promo = "Oui"
            except:
                pass

            # RATING
            rating = "N/A"
            try:
                r = card.find_element(By.CSS_SELECTOR, ".star-rating")
                style = r.get_attribute("style") or ""
                m = re.search(r"width:\s*([\d.]+)%", style)
                if m:
                    rating = str(round(float(m.group(1)) / 20, 1))
            except:
                pass

            # URL
            url_produit = ""
            try:
                url_produit = card.find_element(
                    By.CSS_SELECTOR, "h2.product-title a, a"
                ).get_attribute("href") or ""
            except:
                pass

            if nom and prix_actuel:
                produits.append({
                    "shop":        "Lasolda",
                    "categorie":   categorie,
                    "produit":     nom,
                    "prix":        prix_actuel,
                    "ancien_prix": ancien_prix if ancien_prix else "N/A",
                    "remise_pct":  remise,
                    "stock":       stock,
                    "en_promo":    en_promo,
                    "rating":      rating,
                })

        except:
            continue

    return produits


# ============================================================
# SCRAPER LHMIZA
# Confirmé screenshot 2 :
#   card  → li.item  (dans ul.products-loop.row.grid)
#   nom   → div.item-content.products-content > h4
#   prix  → span.item-price > span > span  (spans imbriqués)
#   remise→ div.sale-off2  (ex: "36%" + "de réduction")
#   promo → div.sale-off.has-newicon contenant "Soldé"
#   stock → classe "instock"/"outofstock" sur li
#   url   → div.item-img > a
# ============================================================
def scrape_cards_lhmiza(driver, categorie):
    produits = []

    try:
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "ul#product_listing, ul.products-loop")
            )
        )
    except:
        print("    ⚠️  Timeout Lhmiza")
        return produits

    # ✅ Cards : li.item dans ul#product_listing (confirmé screenshot)
    cards = driver.find_elements(By.CSS_SELECTOR, "ul#product_listing li.item, li.item")

    for card in cards:
        try:
            # ✅ NOM : div.item-content.products-content > h4 (confirmé)
            nom = ""
            try:
                nom = card.find_element(
                    By.CSS_SELECTOR, "div.item-content.products-content h4"
                ).text.strip()
            except:
                try:
                    nom = card.find_element(By.CSS_SELECTOR, "h4").text.strip()
                except:
                    pass

            # ✅ PRIX : span.item-price contient spans imbriqués (confirmé)
            prix_actuel = ""
            ancien_prix = ""
            try:
                prix_container = card.find_element(By.CSS_SELECTOR, "span.item-price")
                # Récupère tous les spans de prix
                prix_spans = prix_container.find_elements(By.CSS_SELECTOR, "span")
                prix_valides = []
                for s in prix_spans:
                    txt = nettoyer_prix(s.text)
                    if txt and txt not in prix_valides:
                        prix_valides.append(txt)
                if prix_valides:
                    prix_actuel = prix_valides[0]
                if len(prix_valides) > 1:
                    ancien_prix = prix_valides[1]
            except:
                pass

            # ✅ REMISE : div.sale-off2 → "36%" + "de réduction" (confirmé)
            remise = "N/A"
            try:
                remise_txt = card.find_element(
                    By.CSS_SELECTOR, "div.sale-off2"
                ).text.strip()
                m = re.search(r"\d+", remise_txt)
                if m:
                    remise = m.group()
            except:
                # Calcul si les 2 prix disponibles
                try:
                    if prix_actuel and ancien_prix:
                        p_act = float(prix_actuel)
                        p_anc = float(ancien_prix)
                        if p_anc > p_act > 0:
                            remise = str(round((1 - p_act / p_anc) * 100))
                except:
                    pass

            # ✅ PROMO : div.sale-off.has-newicon (confirmé, contient "Soldé")
            en_promo = "Non"
            try:
                badge = card.find_element(
                    By.CSS_SELECTOR, "div.sale-off, div.sw-newlabel"
                ).text.strip().lower()
                if badge:
                    en_promo = "Oui"
            except:
                pass

            # ✅ STOCK : classe sur li.item
            stock = "En stock"
            try:
                classes = card.get_attribute("class") or ""
                if "outofstock" in classes:
                    stock = "Épuisé"
            except:
                pass

            # URL produit
            url_produit = ""
            try:
                url_produit = card.find_element(
                    By.CSS_SELECTOR, "div.item-img a, div.item-detail a"
                ).get_attribute("href") or ""
            except:
                pass

            if nom and prix_actuel:
                produits.append({
                    "shop":        "Lhmiza",
                    "categorie":   categorie,
                    "produit":     nom,
                    "prix":        prix_actuel,
                    "ancien_prix": ancien_prix if ancien_prix else "N/A",
                    "remise_pct":  remise,
                    "stock":       stock,
                    "en_promo":    en_promo,
                    "rating":      "N/A",
                })

        except:
            continue

    return produits


# ============================================================
# PAGINATION — commune aux 2 sites
# ============================================================
def scrape_categorie(driver, shop_name, scraper_fn, categorie, urls):
    produits = []

    for url_base in urls:
        page = 1
        print(f"\n    [{shop_name}] Collection : {url_base}")

        while True:
            url = url_base if page == 1 else f"{url_base.rstrip('/')}/page/{page}/"
            print(f"      Page {page} → {url}")
            driver.get(url)
            time.sleep(2.5)

            # Détection 404 ou redirection
            if page > 1:
                titre = driver.title.lower()
                current = driver.current_url.rstrip("/")
                if "404" in titre or "non trouvé" in titre or current == url_base.rstrip("/"):
                    print(f"      Fin des pages.")
                    break

            nb_avant = len(produits)
            page_produits = scraper_fn(driver, categorie)
            produits.extend(page_produits)

            if not page_produits or len(produits) == nb_avant:
                print(f"      Page vide — fin.")
                break

            # Bouton page suivante
            has_next = False
            try:
                driver.find_element(By.CSS_SELECTOR,
                    "a.next.page-numbers, "
                    ".woocommerce-pagination a.next, "
                    "nav.woocommerce-pagination a[aria-label='Suivant'], "
                    "a[aria-label='Suivant']"
                )
                has_next = True
            except:
                pass

            if not has_next:
                print(f"      Dernière page atteinte.")
                break

            print(f"      ✅ +{len(page_produits)} produits (total : {len(produits)})")
            page += 1
            time.sleep(2)

    return produits


# ============================================================
# MAIN
# ============================================================
def scrape_woocommerce_full():
    driver = setup_driver()
    tous = []
    vus = set()

    scraper_fns = {
        "lasolda": scrape_cards_lasolda,
        "lhmiza":  scrape_cards_lhmiza,
    }

    for shop_name, config in SITES.items():
        fn = scraper_fns[config["scraper"]]
        print(f"\n{'='*55}")
        print(f"  BOUTIQUE : {shop_name}")
        print(f"{'='*55}")

        for categorie, urls in config["categories"].items():
            produits_cat = scrape_categorie(driver, shop_name, fn, categorie, urls)

            for p in produits_cat:
                cle = (p["shop"], p["produit"].lower(), p["prix"])
                if cle not in vus:
                    vus.add(cle)
                    tous.append(p)

            print(f"\n  ✅ {categorie} → {len(produits_cat)} scrapés | {len(tous)} uniques total")
            time.sleep(1.5)

    driver.quit()
    return tous


if __name__ == "__main__":
    resultats = scrape_woocommerce_full()
    df = pd.DataFrame(resultats)
    df.to_csv("dataset_woocommerce.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ {len(df)} produits uniques → dataset_woocommerce.csv")
    print(df.groupby(["shop", "categorie"]).size().to_string())