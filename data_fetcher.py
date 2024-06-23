import requests
from bs4 import BeautifulSoup
import os
import time
import ast


# Function to get image URLs from a search URL
def get_image_urls(search_url, max_images=2000):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    urls = []
    page = 0

    while len(urls) < max_images:
        try:
            # Send a GET request to the search URL with pagination
            response = requests.get(search_url + f'&first={page * 50}', headers=headers, timeout=5)
            response.raise_for_status()  # Check for HTTP errors
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            break

        # Parse the HTML content of the response
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all 'a' tags with class 'iusc' (these contain image data)
        image_elements = soup.find_all('a', class_='iusc')

        if not image_elements:
            break

        for img in image_elements:
            if len(urls) >= max_images:
                break
            m = img.get('m')
            if m:
                try:
                    m = ast.literal_eval(m)
                    if 'murl' in m:
                        urls.append(m['murl'])
                except ValueError:
                    continue

        page += 1
        time.sleep(1)

    return urls


# Function to download images from a list of URLs
def download_images(urls, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for i, url in enumerate(urls):
        try:
            img_data = requests.get(url, timeout=5).content
            image_path = f'{folder_name}/{i + 1}.jpg'
            with open(image_path, 'wb') as handler:
                handler.write(img_data)
        except requests.RequestException as e:
            print(f"Could not download {url}: {e}")


# Example classes and URLs
classes = ['Grizzly Bear', 'Polar Bear', 'American Black Bear', 'Asian Black Bear', 'Sloth Bear']
urls = {
    'Grizzly Bear': 'https://www.bing.com/images/search?q=grizzly+bear+images&form=HDRSC4&first=1',
    'Polar Bear': 'https://www.bing.com/images/search?q=polar+bear+images&form=HDRSC4&first=1',
    'American Black Bear': 'https://www.bing.com/images/search?q=american+black+bear+images&form=HDRSC4&first=1',
    'Asian Black Bear': 'https://www.bing.com/images/search?q=asian+black+bear+images&form=HDRSC4&first=1',
    'Sloth Bear': 'https://www.bing.com/images/search?q=sloth+bear+images&form=HDRSC4&first=1'
}

# Fetch and download images for each class
for cat, url in urls.items():
    print(f"Downloading images for {cat}")
    image_urls = get_image_urls(url, max_images=2000)
    download_images(image_urls, cat)

print("Image downloading complete.")
