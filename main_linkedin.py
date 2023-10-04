import time 
 
 
import pandas as pd 
from selenium import webdriver 
from selenium.webdriver import Chrome 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager

# start by defining the options 
options = webdriver.ChromeOptions() 
options.headless = True # it's more scalable to work in headless mode 
# normally, selenium waits for all resources to download 
# we don't need it as the page also populated with the running javascript code. 

options.page_load_strategy = 'none' 

# maximize the window
options.add_argument("--start-maximized")

# this returns the path web driver downloaded 
chrome_path = ChromeDriverManager().install() 
chrome_service = Service(chrome_path) 

# pass the defined options and service objects to initialize the web driver 
driver = Chrome(options=options, service=chrome_service) 
driver.implicitly_wait(5)

query = 'Software Engineer'
location = 'Singapore'
 
driver.get('https://www.linkedin.com/login')
time.sleep(4)

email_input = driver.find_element(By.ID, 'username')
password_input = driver.find_element(By.ID, 'password')
email_input.send_keys('')
password_input.send_keys('')
# click on login button
driver.find_element(By.XPATH, '//*[@id="organic-div"]/form/div[3]/button').click()
time.sleep(25)

# click on Jobs button
driver.find_element(By.XPATH, '//*[@id="global-nav"]/div/nav/ul/li[3]/a').click()
time.sleep(5)

# Go to search results directly via link with 'Software engineer' and 'Singapore
driver.get('https://www.linkedin.com/jobs/search/?keywords=information%20security&location=Singapore')
time.sleep(7)

# could not obtain job location from the individual job url because it is in plaintext, so crawled the job cards instead
company_locations = []

# Get all job links
links = []
print('Collecting links now.')

try:
    for page in range(2,40):
        # Locate the job section
        jobs_block = driver.find_element(By.CLASS_NAME, 'jobs-search-results-list')
        time.sleep(1)
        # Make an iterable of each job card
        jobs_list = jobs_block.find_elements(By.CSS_SELECTOR, '.jobs-search-results__list-item')

        # Store the url in each job card
        for job in jobs_list:
            # scroll down for each job element
            driver.execute_script('arguments[0].scrollIntoView();', job)

            # store the job locations
            company_locations.append(job.find_element(By.CLASS_NAME, 'job-card-container__metadata-item').text)

            all_links = job.find_elements(By.TAG_NAME, "a")

            for a in all_links:
                if str(a.get_attribute('href')).startswith("https://www.linkedin.com/jobs/view") and a.get_attribute('href') not in links: 
                    links.append(a.get_attribute('href'))
                else:
                    pass

        # go to next page:
        driver.find_element(By.XPATH, f"//button[@aria-label='Page {page}']").click()
        time.sleep(3)

except:
    pass

print(f'Found {len(links)} job links')
print(links)

# Create empty lists to store information
job_titles = []
position_level = []
company_names = []
posting_dates = []
job_desc = []

i=0

for i in range(len(links)):
    try:
        driver.get(links[i])
        i+=1
        print(f'Scraping job link {i}/{len(links)}')
        time.sleep(5)

        # click see more button
        driver.find_element(By.CLASS_NAME, 'artdeco-card__action').click()
        time.sleep(2)

    except:
        pass

    # store job title, position_level, company_name
    contents = driver.find_elements(By.CLASS_NAME, 'p5')
    for content in contents:
        try:
            job_titles.append(content.find_element(By.TAG_NAME, 'h1').text)
            position_level.append(content.find_element(By.CSS_SELECTOR, '.job-details-jobs-unified-top-card__job-insight span').text)
            company_names.append(content.find_element(By.CSS_SELECTOR, '.job-details-jobs-unified-top-card__primary-description .app-aware-link').text)

        except:
            pass


    # store posting_date, job_description
    job_description = driver.find_elements(By.CLASS_NAME, 'jobs-description__content')
    for description in job_description:
        try:
            posting_dates.append(description.find_element(By.CSS_SELECTOR, 'p.t-black--light').text)
            job_desc.append(description.find_element(By.CSS_SELECTOR, '.jobs-description-content__text span').text)

        except:
            pass

print(job_titles)
print(position_level)
print(company_locations)
print(company_names)
print(posting_dates)
print(job_desc)

time.sleep(10)

job_desc[0].replace('\n',' ')

# Creating the dataframe 
df = pd.DataFrame(list(zip(job_titles, position_level, company_locations, company_names, job_desc, posting_dates, links)),
                    columns =['job_title', 'position_level',
                           'company_location','company_name',
                           'job_desc','date_posted', 'url'])

# Storing the data to csv file
df.to_csv('job_list_IS.csv', index=False)