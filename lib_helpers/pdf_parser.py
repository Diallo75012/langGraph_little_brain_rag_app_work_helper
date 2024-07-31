# PDF load and then parse the file to get what is structured and usable by llm for embedding/retrival but only the quality data, the rest will need to be searched online using some other tools

import pandas as pd
import PyPDF2
import re
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

# Function to extract text and sections from a PDF
def pdf_to_sections(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extract_text()
    
    lines = text.split('\n')
    data = []
    current_section = None
    section_buffer = []

    def is_title(line):
        return bool(re.match(r'^(\d+[\.\-]?\s+|Chapter|Section|[\dA-Z]\.\s+)', line)) or len(line) < 50

    def is_bullet_point(line):
        return bool(re.match(r'^(\-|\*|\+|\d+\.)\s', line))

    def is_table_line(line):
        return bool(re.match(r'^\|.*\|$', line))

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for Titles, Headings, Subheadings, etc.
        if is_title(line):
            if section_buffer:
                data.append({
                    'text': ' '.join(section_buffer),
                    'section': current_section,
                    'document': pdf_path
                })
                section_buffer = []
            current_section = line
        elif is_bullet_point(line):
            section_buffer.append(f"Bullet Point: {line}")
        elif is_table_line(line):
            section_buffer.append(f"Table Row: {line}")
        else:
            section_buffer.append(line)
    
    # Add any remaining buffer to data
    if section_buffer:
        data.append({
            'text': ' '.join(section_buffer),
            'section': current_section,
            'document': pdf_path
        })

    return data

# Parse the PDF and create a DataFrame
pdf_data = pdf_to_sections('example.pdf')
df = pd.DataFrame(pdf_data)

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
df['doc_name'] = df['document'].apply(lambda x: os.path.basename(x))
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

