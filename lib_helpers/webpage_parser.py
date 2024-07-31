# # Webpage load and then parse the file to get what is structured and usable by llm for embedding/retrival but only the quality data, the rest will need to be searched online using some other tools

### Extract and Parse Web Page Content
import requests
from bs4 import BeautifulSoup
import pandas as pd
import psycopg2
from uuid import uuid4
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatGroq

load_dotenv()

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
def web_page_to_sections(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    data = []
    current_section = None
    current_title = None

    for element in soup.find_all(['h1', 'h2', 'h3', 'p']):
        if element.name in ['h1', 'h2', 'h3']:
            current_title = element.get_text(strip=True)
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text:
                data.append({'text': text, 'section': current_title, 'url': url})
    
    return data

# Parse the web page and create a DataFrame
web_data = web_page_to_sections('https://example.com')
df = pd.DataFrame(web_data)

# Initialize the Groq LLM for summarization
groq_llm_mixtral_7b = ChatGroq(
    temperature=float(os.getenv("GROQ_TEMPERATURE")), 
    groq_api_key=os.getenv("GROQ_API_KEY"), 
    model_name=os.getenv("MODEL_MIXTRAL_7B"),
    max_tokens=int(os.getenv("GROQ_MAX_TOKEN"))
)

# Function to summarize text using Groq LLM
def summarize_text(row):
    response = groq_llm_mixtral_7b(row['text'], max_length=50, min_length=25, do_sample=False)
    return response['choices'][0]['text'].strip()

# Apply summarization
df['summary'] = df.apply(summarize_text, axis=1)

# Generate metadata and add UUID and retrieved fields
df['id'] = [str(uuid4()) for _ in range(len(df))]
df['doc_name'] = df['url'].apply(lambda x: x)
df['title'] = df['section']
df['content'] = df['summary']
df['retrieved'] = False

# Select only the necessary columns
df_final = df[['id', 'doc_name', 'title', 'content', 'retrieved']]

# Function to store cleaned data in PostgreSQL
def store_data(df, conn):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT INTO documents (id, doc_name, title, content, retrieved) VALUES (%s, %s, %s, %s, %s)",
            (row['id'], row['doc_name'], row['title'], row['content'], row['retrieved'])
        )
    conn.commit()
    cursor.close()

# Store the cleaned data in PostgreSQL
conn = connect_db()
store_data(df_final, conn)
conn.close()


