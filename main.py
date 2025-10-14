# ============================================================
# Extraction de données de crypto-monnaies avec Selenium
# Auteur : KELLA OMAR
# Inspiré du dépôt Praveen76 / Web-Scraping-using-Selenium-Python
# ============================================================

# --- Importation des bibliothèques nécessaires ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os


chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")

# 👇 Cette ligne télécharge et configure automatiquement le bon driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# --- Étape 2 : Accès au site CoinMarketCap ---
url = "https://coinmarketcap.com/"
driver.get(url)

print("Chargement de la page CoinMarketCap...")
time.sleep(5)  # temps d’attente pour le chargement complet de la page

# --- Étape 3 : Extraction des lignes du tableau ---
# Chaque crypto est dans une balise <tr> du tableau principal
rows = driver.find_elements(By.XPATH, '//tbody/tr')

cryptos_data = []

# --- Étape 4 : Parcours et extraction des informations ---
for row in rows[:20]:  # on limite à 20 premières cryptos
    try:
#name = row.find_element(By.XPATH, './td[3]//p[contains(@class,"coin-item-symbol")]').text
#Soooooooooooo :
#row → correspond à une ligne <tr> du tableau.
#find_element → cherche un seul élément à l’intérieur de cette ligne.
#By.XPATH → indique qu’on utilise un sélecteur XPath pour repérer l’élément.
#'./td[3]//p[contains(@class,"coin-item-symbol")]' :
#./td[3] → va dans la 3ᵉ cellule (<td>) de la ligne.
#//p[contains(@class,"coin-item-symbol")] → cherche un paragraphe <p> dont la classe contient le mot coin-item-symbol.
#.text → récupère le texte à l’intérieur de cette balise (ex. "BTC").

                                                         
        name = row.find_element(By.XPATH, './td[3]//p[contains(@class,"coin-item-symbol")]').text
        full_name = row.find_element(By.XPATH, './td[3]//p[contains(@class,"coin-item-name")]').text
        price = row.find_element(By.XPATH, './td[4]').text
        change_24h = row.find_element(By.XPATH, './td[5]').text
        volume_24h = row.find_element(By.XPATH, './td[7]').text
        market_cap = row.find_element(By.XPATH, './td[8]').text

        cryptos_data.append({
            "Nom": full_name,
            "Symbole": name,
            "Prix": price,
            "Variation_24h": change_24h,
            "Volume_24h": volume_24h,
            "Capitalisation": market_cap
        })
    except Exception as e:
        print(f"Erreur lors de l’extraction d’une ligne : {e}")

# --- Étape 5 : Fermeture du navigateur ---
driver.quit()

# --- Étape 6 : Sauvegarde dans un fichier CSV ---
if not os.path.exists("data"):
    os.mkdir("data")

df = pd.DataFrame(cryptos_data)
df.to_csv("data/cryptos_data.csv", index=False, encoding="utf-8-sig")

print("\n Extraction terminée avec succès !")
print("Données enregistrées dans : data/cryptos_data.csv")

# --- Étape 7 : Aperçu des données ---
print(df.head(10))
