import requests
from bs4 import BeautifulSoup
import csv
import panda as pd
import time
from fake_useragent import UserAgent

headers = {'User-Agent':UserAgent().random}


products=[]
urls = ["https://sg.indeed.com/jobs?q=&l=Singapore&from=searchOnHP&vjk=ba617eea174b9109"]
visited_urls = []
while len(urls) != 0:
    current_url = urls.pop()

    response = requests.get(current_url, headers)
    soup = BeautifulSoup(response.content, "html.parser")
    job_listings = soup.find_all('div', class_="jobsearch-SerpJobCard")
    job_data = []
    for listing in job_listings:
        title = listing.find('a', class_='jobtitle').text.strip()
    df = pd.DataFrame(job_Data, columns = ['Title'])

    df.to_csv('indeed_sg_jobs.csv', index=False)
    visited_urls.append(current_url)

    link_elements = soup.select("a[href]")

    for link_element in link_elements:
        url = link_element['href']
        if "https://sg.indeed.com" in url:
            if url not in visited_urls and url not in urls:
                urls.append(url)


    data = []
    for item in soup.find_all('div', id_='mosaic-provider-jobcards'):
        data.append(item)
    print(data)


    # products.append(product)
    with open('products.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)

        for product in products:
            writer.writerow(product.values())

    print(visited_urls)
