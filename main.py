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
import os.path
from bs4 import BeautifulSoup

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'

# Selenium options (Chromedriver)
options = webdriver.ChromeOptions()
options.add_argument(f'user-agent={user_agent}')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')  # Fresh browser
options.add_argument('--blink-settings=imagesEnabled=false') # Disable images, faster loading
# options.add_argument('--headless=new')
options.page_load_strategy = 'none'
chrome_path = ChromeDriverManager().install()
chrome_service = Service(chrome_path)
driver = Chrome(options=options, service=chrome_service)
driver.implicitly_wait(5)

job_data = []
links = []

# Declare CSV File Name and Headings
csvFileName = "indeed_jobs.csv"
csvHeading = ['jobTitle', 'jobCompany', 'jobLocation', 'jobMeta', 'jobContent', 'url']

if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline="") as newcsv:
        writer = csv.DictWriter(newcsv, fieldnames=csvHeading)
        writer.writeheader()


job_title = 'software engineer'
location = 'Singapore'
url = "https://sg.indeed.com/jobs?q="+job_title+"&l="+location+""
driver.get(url)
driver.implicitly_wait(3)

pages = 50

# Retrieve links
for i in range(pages):
    job_listings = driver.find_elements(By.CLASS_NAME, 'jcs-JobTitle')  # Customize the XPath as needed
    print(job_listings)
    for listing in job_listings:
        driver.implicitly_wait(15)
        product_url = listing.get_attribute('href')
        links.append(product_url)
        print(links)
    if driver.find_elements(By.XPATH, "//a[@data-testid='pagination-page-next']"):
        element = driver.find_element(By.XPATH, "//a[@data-testid='pagination-page-next']")
        driver.execute_script("arguments[0].click();", element)
        print("clicking on next page")
    else:  # If unable to click "next"
        print("Could not click next page, reached last page. Scraping jobs now...")
        break

# Go thru links and scrape
for link in links:
    driver.get(link)
    driver.implicitly_wait(15)

    try:
        title = driver.find_element(By.CLASS_NAME, 'jobsearch-JobInfoHeader-title-container').text.strip()
    except:
        try:
            title = driver.find_element(By.CLASS_NAME, 'jobsearch-JobInfoHeader-title-container').text.strip()
        except:
            title = "Not Found"

    try:
        job_link = link
    except:
        link = "Not Found"

    try:
        company = driver.find_element(By.XPATH, '//div[@data-testid="inlineHeader-companyName"]').text.strip()
    except:
        company = "Not Found"

    try:
        location = driver.find_element(By.XPATH, '//div[@data-testid="inlineHeader-companyLocation"]').text.strip()
    except:
        location = "Not Found"

    try:
        description = driver.find_element(By.ID, 'jobDescriptionText').text.strip()
    except:
        description = "Not Found"

    try:
        meta = driver.find_element(By.ID, 'salaryInfoAndJobType').text.strip()
    except:
        meta = "Not Found"


    job_data.append({'Job Title': title, 'Tags': meta, 'Company': company, 'Location':location, 'Description':description})
    print(f"\nTitle: {title}\nLink:{job_link}\nCompany:{company}\nMeta:{meta}\nDescription:{description}\nLocation:{location}\n")
    driver.implicitly_wait(3)

    reviewDictEntry = {
        'jobTitle': title,
        'jobCompany': company,
        'jobLocation': location,
        'jobMeta': meta,
        'url': driver.current_url,
        'jobContent': description
    }

    with open(csvFileName, 'a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csvHeading)
        w.writerow(reviewDictEntry)

driver.quit()

