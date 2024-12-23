# %%
# !pip install BeautifulSoup

# %%
import os
import json
import time
import random
import zipfile
import requests
import pandas as pd
from bs4 import BeautifulSoup

# %% [markdown]
# # Class Explanation: `NewsScraper`
# 
# ## Overview
# The `NewsScraper` class is designed for scraping news articles from three different Urdu news websites: Geo, Jang, and Express. The class has methods that cater to each site's unique structure and requirements. Below, we will go through the class and its methods, detailing what each function does, the input it takes, and the output it returns.
# 
# ## Class Definition
# 
# ```python
# class NewsScraper:
#     def __init__(self, id_=0):
#         self.id = id_
# ```
# 
# 
# ## Method 1: `get_express_articles`
# 
# ### Description
# Scrapes news articles from the Express website across categories like saqafat (entertainment), business, sports, science-technology, and world. The method navigates through multiple pages for each category to gather a more extensive dataset.
# 
# ### Input
# - **`max_pages`**: The number of pages to scrape for each category (default is 7).
# 
# ### Process
# - Iterates over each category and page.
# - Requests each category page and finds article cards within `<ul class='tedit-shortnews listing-page'>`.
# - Extracts the article's headline, link, and content by navigating through `<div class='horiz-news3-caption'>` and `<span class='story-text'>`.
# 
# ### Output
# - **Returns**: A tuple of:
#   - A Pandas DataFrame containing columns: `id`, `title`, and `link`).
#   - A dictionary `express_contents` where the key is the article ID and the value is the article content.
# 
# ### Data Structure
# - Article cards are identified by `<li>` tags.
# - Content is structured within `<span class='story-text'>` and `<p>` tags.
# 
# 

# %%
class NewsScraper:
    def __init__(self,id_=0):
        self.id = id_

  # write functions to scrape from other websites

    def get_express_articles(self, max_pages=14):
        express_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
            "news_channel": [], # optional
        }
        base_url = 'https://www.express.pk'
        categories = ['saqafat', 'business', 'sports', 'science', 'world']   # saqafat is entertainment category

        # Iterating over the specified number of pages
        for category in categories:
            for page in range(1, max_pages + 1):
                print(f"Scraping page {page} of category '{category}'...")
                url = f"{base_url}/{category}/archives?page={page}"
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Finding article cards
                cards = soup.find('ul', class_='tedit-shortnews listing-page').find_all('li')  # Adjust class as per actual site structure
                print(f"\t--> Found {len(cards)} articles on page {page} of '{category}'.")

                success_count = 0

                for card in cards:
                    try:
                        div = card.find('div',class_='horiz-news3-caption')

                        # Article Title
                        headline = div.find('a').get_text(strip=True).replace('\xa0', ' ')

                        # Article link
                        link = div.find('a')['href']

                        # Requesting the content from each article's link
                        article_response = requests.get(link)
                        article_response.raise_for_status()
                        content_soup = BeautifulSoup(article_response.text, "html.parser")


                        # Content arranged in paras inside <span> tags
                        paras = content_soup.find('span',class_='story-text').find_all('p')

                        combined_text = " ".join(
                        p.get_text(strip=True).replace('\xa0', ' ').replace('\u200b', '')
                        for p in paras if p.get_text(strip=True)
                        )

                        # Storing data
                        express_df['id'].append(self.id)
                        express_df['title'].append(headline)
                        express_df['link'].append(link)
                        express_df['gold_label'].append(category.replace('saqafat','entertainment').replace('science','science-technology'))
                        express_df['content'].append(combined_text)
                        express_df["news_channel"].append("Express News")  # Optional

                        # Increment ID and success count
                        self.id += 1
                        success_count += 1

                    except Exception as e:
                        print(f"\t--> Failed to scrape an article on page {page} of '{category}': {e}")

                print(f"\t--> Successfully scraped {success_count} articles from page {page} of '{category}'.")
            print('')

        return pd.DataFrame(express_df)
    
    
    
    def get_dunya_articles(self, max_pages=14):
        dunya_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
            "news_channel": [], # optional
        }
        base_url = 'https://urdu.dunyanews.tv'
        categories = ['Entertainment', 'Pakistan', 'World', 'Sports', 'Business']

        for category in categories:
            for page in range(1, max_pages + 1):
                print(f"Scraping page {page} of category '{category}'...")
                url = f"{base_url}/index.php/ur/{category}?page={page}"
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                news_boxes = soup.find_all("div", class_="cNewsBox")
                print(f"\t--> Found {len(news_boxes)} articles on page {page} of '{category}'.")

                success_count = 0

                for news in news_boxes:
                    try:
                        title_tag = news.find("h3")
                        if title_tag:
                            link_tag = title_tag.find("a")
                            if link_tag:
                                title = link_tag.get_text(strip=True)
                                link = base_url + link_tag['href']
                            else:
                                print("\t--> Skipping article due to missing link.")
                                continue
                        else:
                            print("\t--> Skipping article due to missing title tag.")
                            continue

                        article_response = requests.get(link)
                        article_response.raise_for_status()
                        content_soup = BeautifulSoup(article_response.text, "html.parser")

                        content = ""
                        content_div = content_soup.find("div", class_="main-news") 
                        if content_div:
                            content_paras = content_div.find_all("p")
                            content = " ".join(
                                p.get_text(strip=True).replace('\xa0', ' ').replace('\u200b', '')
                                for p in content_paras if p.get_text(strip=True)
                            )

                        if not content:
                            print(f"\t--> Skipping article '{title}' due to missing content.")
                            continue

                        dunya_df['id'].append(self.id)
                        dunya_df['title'].append(title)
                        dunya_df['link'].append(link)
                        dunya_df['gold_label'].append(category)
                        dunya_df['content'].append(content)
                        dunya_df["news_channel"].append("Dunya News")  # Optional

                        self.id += 1
                        success_count += 1

                    except Exception as e:
                        print(f"\t--> Failed to scrape article due to: {e}")

                print(f"\t--> Successfully scraped {success_count} articles from page {page} of '{category}'.")
            print('')

        return pd.DataFrame(dunya_df)



    def get_geo_articles(self, max_pages=14):
        geo_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
            "news_channel": [], # optional
        }
        base_url = 'https://urdu.geo.tv/category'
        categories = ['business', 'entertainment', 'sports', 'world']

        for category in categories:
            for page in range(1, max_pages + 1):
                print(f"Scraping page {page} of category '{category}'...")
                url = f"{base_url}/{category}/page/{page}/"
                response = requests.get(url)
                if response.status_code == 403:
                    print("Request was blocked by the server.")
                    break
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                articles = soup.find_all("div", class_="m_pic")

                print(f"\t--> Found {len(articles)} articles on page {page} of '{category}'.")

                success_count = 0
                for article in articles:
                    try:
                        title_tag = article.find("a", class_="open-section")
                        if title_tag:
                            title = title_tag.get("title", "").strip()
                            link = title_tag["href"]
                        else:
                            print("\t--> Skipping article due to missing title or link.")
                            continue

                        article_response = requests.get(link)
                        article_response.raise_for_status()
                        content_soup = BeautifulSoup(article_response.text, "html.parser")
                        
                        content_div = content_soup.find("div", class_="content-area")
                        content = ""
                        if content_div:
                            content = " ".join(
                                p.get_text(strip=True)
                                for p in content_div.find_all("p")
                            )

                        if not content:
                            print(f"\t--> Skipping article '{title}' due to missing content.")
                            continue

                        geo_df["id"].append(self.id)
                        geo_df["title"].append(title)
                        geo_df["link"].append(link)
                        geo_df["gold_label"].append(category.capitalize())
                        geo_df["content"].append(content)
                        geo_df["news_channel"].append("Geo News") 

                        self.id += 1
                        success_count += 1

                    except Exception as e:
                        print(f"\t--> Failed to scrape article due to: {e}")

                print(f"\t--> Successfully scraped {success_count} articles from page {page} of '{category}'.")
            print('')

        return pd.DataFrame(geo_df)



    def get_jang_articles(self):
        jang_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
            "news_channel": [], # optional
        }

        base_url = 'https://jang.com.pk/category/latest-news'
        categories = ['entertainment', 'sports', 'world', 'health-science']

        for category in categories:
            print(f"Scraping category '{category}'...")
            url = f"{base_url}/{category}/"
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            articles = soup.find('ul',class_='scrollPaginationNew__').find_all("li")
            print(f"\t--> Found {len(articles)} articles in '{category}'.")

            success_count = 0
            for article in articles:
                try:
                    if article.get("class") == ["ad_latest_stories"]:
                        continue
                    
                    title, link = None, None

                    main_pic = article.find("div", class_="main-pic")
                    if main_pic:
                        link_tag = main_pic.find("a", href=True)
                        if link_tag:
                            link = link_tag["href"]
                            title = link_tag.get("title", "").strip()
                            print(title)

                    if not title or not link:
                        main_heading = article.find("div", class_="main-heading")
                        if main_heading:
                            link_tag = main_heading.find("a", href=True)
                            if link_tag:
                                link = link_tag["href"]
                                title_tag = link_tag.find("h2")
                                title = title_tag.get_text(strip=True) if title_tag else ""

                    if not title or not link:
                        print("\t--> Skipping article due to missing title or link.")
                        continue

                    article_response = requests.get(link)
                    article_response.raise_for_status()
                    content_soup = BeautifulSoup(article_response.text, "html.parser")

                    content_div = content_soup.find("div", class_="detail_view_content")
                    content = ""
                    if content_div:
                        content = " ".join(
                            p.get_text(strip=True)
                            for p in content_div.find_all("p")
                        )

                    if not content:
                        print(f"\t--> Skipping article '{title}' due to missing content.")
                        continue

                    jang_df["id"].append(self.id)
                    jang_df["title"].append(title)
                    jang_df["link"].append(link)
                    jang_df["gold_label"].append(category.capitalize())
                    jang_df["content"].append(content)
                    jang_df["news_channel"].append("Jang")  

                    self.id += 1
                    success_count += 1

                except Exception as e:
                    print(f"\t--> Failed to scrape article due to: {e}")

            print(f"\t--> Successfully scraped {success_count} articles from '{category}'.")

        return pd.DataFrame(jang_df)



    def get_dawn_articles(self):
        dawn_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
            "news_channel": [], # optional
        }
        base_url = 'https://www.dawnnews.tv/'
        categories = ['business','sport', 'tech', 'world']

        for category in categories:
            print(f"Scraping category '{category}'...")
            url = f"{base_url}/{category}/"
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            articles = soup.find('div',class_='flex flex-row w-auto').find_all("article")
            print(f"\t--> Found {len(articles)} articles in '{category}'.")

            success_count = 0
            for article in articles:
                try:
                    title, link = None, None

                    main_div = article.find("h2", class_="story__title")
                    if main_div:
                        link_tag = main_div.find("a", href=True)
                        if link_tag:
                            link = link_tag["href"]
                            title = link_tag.text.strip()
                            print(title)

                    if not title or not link:
                        print("\t--> Skipping article due to missing title or link.")
                        continue

                    article_response = requests.get(link)
                    article_response.raise_for_status()
                    content_soup = BeautifulSoup(article_response.text, "html.parser")

                    content_div = content_soup.find("div", class_="story__content")
                    content = ""
                    if content_div:
                        content = " ".join(
                            p.get_text(strip=True)
                            for p in content_div.find_all("p")
                        )

                    if not content:
                        print(f"\t--> Skipping article '{title}' due to missing content.")
                        continue

                    dawn_df["id"].append(self.id)
                    dawn_df["title"].append(title)
                    dawn_df["link"].append(link)
                    dawn_df["gold_label"].append(category.capitalize())
                    dawn_df["content"].append(content)
                    dawn_df["news_channel"].append("Dawn News") 

                    self.id += 1
                    success_count += 1

                except Exception as e:
                    print(f"\t--> Failed to scrape article due to: {e}")

            print(f"\t--> Successfully scraped {success_count} articles from '{category}'.")

        return pd.DataFrame(dawn_df)


# %%
scraper = NewsScraper()

# %%
express_df = scraper.get_express_articles()
dunya_df = scraper.get_dunya_articles()
geo_df = scraper.get_geo_articles()
jang_df = scraper.get_jang_articles()
dawn_df = scraper.get_dawn_articles()

# %% [markdown]
# # Output
# - Save a combined csv of all 5 sites.

# %%
combined_df = pd.concat([jang_df,geo_df,dunya_df,express_df,dawn_df], ignore_index=True)
combined_df.to_csv('FINAL_DATASET.csv', index=False)


