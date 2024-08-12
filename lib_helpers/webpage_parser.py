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
def summarize_text(llm, row, maximum):
    # Add length constraints directly to the prompt
    prompt_text = f"Please summarize the following text in no more than {maximum} characters:\n\n{row['text']}. Make sure  your answering in markdown format putting the answer between ```python ```."
    
    response = llm.invoke(prompt_text)
    print("LLM RESPONSE: ", response)
    summary = response.content.split("```")[1].strip("python").strip()
    #summary = response['choices'][0]['text'].strip()
    print("Summary: ", summary)
    return summary

def generate_title(llm, row, maximum):
    prompt_text = f"Please create a title in no more than {maximum} characters for:\n\n{row['section']} - {row['text']}.Make sure  your answering in markdown format putting the answer between ```python ```."
    response = llm.invoke(prompt_text)
    title = response.content.split("```")[1].strip("python").strip()
    return title

def get_final_df(llm: ChatGroq, webpage_url: str, maximum_content_length: int, maximum_title_length: int):
  # parse webpage content
  webpage_data = web_page_to_sections(webpage_url)
  # put content in a pandas dataframe
  df = pd.DataFrame(webpage_data)
  print("DF: ", df)

  df['summary'] = df.apply(lambda row: summarize_text(llm, row, maximum_content_length), axis=1) # here use llm to make the summary

  # Generate titles using section and text
  df['title'] = df.apply(lambda row: generate_title(llm, row, maximum_title_length), axis=1)

  # Generate metadata and add UUID and retrieved fields
  df['id'] = [str(uuid4()) for _ in range(len(df))]
  df['doc_name'] = df['url']
  df['content'] = df['summary']
  df['retrieved'] = False

  # Select only the necessary columns
  df_final = df[['id', 'doc_name', 'title', 'content', 'retrieved']]
  print("Df Final from library: ", df_final, type(df_final))
  with open("test_url_parser_df_output.csv", "w", encoding="utf-8") as f:
    df_final.to_csv(f, index=False)
  return df_final
