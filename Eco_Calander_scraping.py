from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=options)

url = "https://www.investing.com/economic-calendar/"
driver.get(url)
time.sleep(5)
try:
    accept_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Accept')]")
    accept_button.click()
    time.sleep(3)
except:
    print('pas de cookies')

events = driver.find_elements(By.CLASS_NAME, "js-event-item")
data = []
for event in events:
    try:
        time_ = event.find_element(By.CLASS_NAME, "time").text
        currency = event.find_element(By.CLASS_NAME, "left.flagCur.noWrap").text
        event_name = event.find_element(By.CLASS_NAME, "event").text
        actual = event.find_element(By.CLASS_NAME, "act").text
        forcast = event.find_element(By.CLASS_NAME, "fore").text
        previous = event.find_element(By.CLASS_NAME, "prev").text
        data.append([time_, currency, event_name, actual, forcast, previous])
    except:
        continue
print(data)

df = pd.DataFrame(data, columns=["Heure", "Devise", "Evenement", "Actuel", "Prévision", "Précedent"])
df.to_csv("calandrier_eco.csv", index=False, encoding="utf-8")
driver.quit()
df = pd.read_csv('calandrier_eco.csv')
print(df)