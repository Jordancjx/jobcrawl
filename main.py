import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'

options = webdriver.ChromeOptions()
options.add_argument(f'user-agent={user_agent}')
options.headless = True
# options.add_argument('--headless=new')
options.page_load_strategy = 'none'
chrome_path = ChromeDriverManager().install()
chrome_service = Service(chrome_path)
driver = Chrome(options=options, service=chrome_service)
driver.implicitly_wait(5)

job_data = []

job_title = 'chef'
location = 'Singapore'
url = "https://sg.indeed.com/jobs?q="+job_title+"&l="+location+""
driver.get(url)
driver.implicitly_wait(3)
job_listings = driver.find_elements(By.CLASS_NAME, 'jcs-JobTitle')  # Customize the XPath as needed
print(job_listings)


# Write the header row
for listing in job_listings:
    print(listing)
    driver.implicitly_wait(15)
    product_url = listing.get_attribute('href')
    driver.get(product_url)
    driver.implicitly_wait(5)
    revealed = driver.find_element(By.CLASS_NAME, "fastviewjob")
    WebDriverWait(driver, timeout=2).until(lambda d: revealed.is_displayed())
    driver.implicitly_wait(10)
    title = driver.find_element(By.CLASS_NAME, 'jobsearch-JobInfoHeader-title').text.strip()
    company = driver.find_element(By.XPATH, '//div[@data-testid="inlineHeader-companyName"]').text.strip()
    # location = driver.find_element(By.XPATH, '//div[@data-testid="inlineHeader-companyLocation').text.strip()
    description = driver.find_element(By.ID, 'jobDescriptionText').text.strip()
    meta = driver.find_element(By.ID, 'jobDetailsSection').text.strip()
    job_data.append({'Job Title': title, 'Tags': meta, 'Company': company, 'Location':location, 'Description':description})
    print(f"Tags:{meta}\nTitle: {title}\nCompany: {company}\nLocation: {location}\n")
    driver.execute_script("window.history.go(-1)")
    driver.implicitly_wait(3)



driver.quit()

df = pd.DataFrame(job_data)
csv_filename = 'job_data.csv'
df.to_csv(csv_filename, index=False)
