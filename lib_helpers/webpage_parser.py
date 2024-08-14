# # Webpage load and then parse the file to get what is structured and usable by llm for embedding/retrival but only the quality data, the rest will need to be searched online using some other tools

### Extract and Parse Web Page Content
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from typing import Dict, Any, List, Optional


load_dotenv()

# Function to extract text and sections from a web page
def scrape_website(url: str) -> Dict[str, Any]:
    """"Will get the content of webpage and return a dictionary with keys `text`, `section`, `url`"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Define how to find the title and the main content
    title_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']  # Modify this depending on the site structure
    content_tags = ['p', 'li']  # Modify this depending on the site structure

    # Extract the page title
    current_title = soup.title.string if soup.title else "No Title"

    # Extract all text content
    text = []
    for tag in content_tags:
        elements = soup.find_all(tag)
        for element in elements:
            text.append(element.get_text())

    # Combine all the text
    text = " ".join(text)

    # Prepare the result as a dictionary
    result = {
        'text': text.strip(),
        'section': current_title.strip(),
        'url': url
    }

    return result

