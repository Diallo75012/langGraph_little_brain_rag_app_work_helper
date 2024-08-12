# # Webpage load and then parse the file to get what is structured and usable by llm for embedding/retrival but only the quality data, the rest will need to be searched online using some other tools

### Extract and Parse Web Page Content
import requests
from bs4 import BeautifulSoup
import pandas as pd
import psycopg2
from uuid import uuid4
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from lib_helpers.chunking_module import create_chunks_from_data
from typing import Dict, Any, List, Optional

load_dotenv()

# Initialize the Groq LLM for summarization
groq_llm_mixtral_7b = ChatGroq(
    temperature=float(os.getenv("GROQ_TEMPERATURE")), 
    groq_api_key=os.getenv("GROQ_API_KEY"), 
    model_name=os.getenv("MODEL_MIXTRAL_7B"),
    max_tokens=int(os.getenv("GROQ_MAX_TOKEN"))
)

# Connect to the PostgreSQL database
def connect_db() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        database=os.getenv("DATABASE"), 
        user=os.getenv("USER"), 
        password=os.getenv("PASSWORD"),
        host=os.getenv("HOST"), 
        port=os.getenv("PORT")
    )

def create_table_if_not_exists():
    """Create the documents table if it doesn't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY,
            doc_name TEXT,
            title TEXT,
            content TEXT,
            retrieved BOOLEAN
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

create_table_if_not_exists()

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

