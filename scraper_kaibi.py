from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

def setup_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless") # Désactivez pour voir le navigateur
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def scrape_kiabi_all():
    driver = setup_driver()
    
    # 1. Dictionnaire des catégories avec leurs URLs réelles sur Kiabi.ma
    categories = {
        "Femme": "https://kiabi.ma/collections/k-femme",
        "Homme": "https://kiabi.ma/collections/k-homme",
        "Fille": "https://kiabi.ma/collections/k-fille",
        "Garçon": "https://kiabi.ma/collections/k-garcon",
        "Bébé": "https://kiabi.ma/collections/tout-pour-bebe",
        "Puériculture": "https://kiabi.ma/collections/k-puericulture",
        "Chaussures": "https://kiabi.ma/collections/k-chaussures"
    }

    tous_les_produits = []

    for nom_cat, url in categories.items():
        print(f"--- Début du scraping : {nom_cat} ---")
        driver.get(url)
        
        # On essaie de récupérer 300 produits par catégorie pour atteindre ton quota total
        compteur_cat = 0
        while compteur_cat < 300:
            try:
                # Attente du chargement de la grille (id="product-grid")
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "product-grid")))
                
                # Récupération des cartes produits (div.product-card)
                cards = driver.find_elements(By.CSS_SELECTOR, ".product-card")
                
                for card in cards:
                    try:
                        nom = card.find_element(By.CSS_SELECTOR, "h3 span.capitalize").text
                        prix_brut = card.find_element(By.CSS_SELECTOR, "span.text-primary").text
                        
                        # Nettoyage du prix pour le Data Mining (ex: "275 dh" -> 275)
                        prix_num = prix_brut.lower().replace('dh', '').replace(' ', '').strip()

                        tous_les_produits.append({
                            "shop": "Kiabi",
                            "categorie": nom_cat,
                            "produit": nom,
                            "prix": prix_num,
                            "stock": "En stock",
                            "rating": 4.2 # Note moyenne fictive pour le scoring ML
                        })
                        compteur_cat += 1
                    except:
                        continue

                # PAGINATION : On cherche le bouton "Suivant"
                try:
                    btn_next = driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
                    driver.execute_script("arguments[0].click();", btn_next)
                    time.sleep(4) # Pause importante pour le chargement
                except:
                    print(f"Fin des pages pour {nom_cat}")
                    break
            except Exception as e:
                print(f"Erreur sur la catégorie {nom_cat}: {e}")
                break

    driver.quit()
    return tous_les_produits

# --- LANCEMENT ET SAUVEGARDE ---
resultats = scrape_kiabi_all()
df = pd.DataFrame(resultats)

# Sauvegarde au format CSV (encodage utf-8-sig pour Excel)
df.to_csv("dataset_kiabi_final.csv", index=False, encoding='utf-8-sig')
print(f"Terminé ! {len(df)} produits enregistrés dans dataset_kiabi_final.csv")