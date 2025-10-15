# file.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
import os
import time

# --- Configuration du navigateur ---
options = Options()
options.add_argument("--headless")  # mode sans interface
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Démarrer le navigateur Chrome
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# URL cible
url = "https://unsplash.com/s/photos/forest"
driver.get(url)
time.sleep(5)  # Attendre le chargement de la page

# --- Scroller pour charger plus d'images (optionnel) ---
scroll_pause_time = 2
for i in range(3):  # scroller 3 fois
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)

# --- Extraire les URLs des images ---
images = driver.find_elements(By.TAG_NAME, "img")
image_urls = []

for img in images:
    src = img.get_attribute("src")
    if src and "images.unsplash.com" in src:
        image_urls.append(src)

driver.quit()

# --- Créer un dossier pour stocker les images ---
os.makedirs("forest_images", exist_ok=True)

# --- Télécharger les images ---
print(f"{len(image_urls)} images trouvées.")
for i, link in enumerate(image_urls):
    try:
        img_data = requests.get(link).content
        filename = f"forest_images/forest_{i+1}.jpg"
        with open(filename, "wb") as f:
            f.write(img_data)
        print(f"Téléchargé : {filename}")
    except Exception as e:
        print(f"Erreur pour l'image {i+1}: {e}")

print("Téléchargement terminé !")
