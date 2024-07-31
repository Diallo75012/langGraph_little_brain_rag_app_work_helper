# PDF load and then parse the file to get what is structured and usable by llm for embedding/retrival but only the quality data, the rest will need to be searched online using some other tools

import pandas as pd
import PyPDF2
import re


### Load and Parse Documents with Section Identification

# Function to extract text and sections from a PDF
def pdf_to_sections(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extract_text()
    
    # Split the text into sections and paragraphs
    sections = re.split(r'\n\s*\n', text)  # Simple split based on double newlines
    data = []
    current_section = None
    for section in sections:
        if section.strip().startswith('Chapter') or re.match(r'^Section \d+', section.strip()):
            current_section = section.strip()
        else:
            paragraphs = section.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    data.append({'text': paragraph.strip(), 'section': current_section, 'document': pdf_path})
    return data

# Parse the PDF and create a DataFrame
pdf_data = pdf_to_sections('example.pdf')
df = pd.DataFrame(pdf_data)

### Generate Columns with LLM Analysis and Enhance DataFrame

from transformers import pipeline

# Load an LLM model for analysis (using transformers as an example)
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
summarizer = pipeline('summarization')

# Function to classify text relevance
def classify_text(row):
    result = classifier(row['text'], candidate_labels=['relevant', 'irrelevant'])
    return result['labels'][0]

# Apply classification
df['classification'] = df.apply(classify_text, axis=1)

# Judge relevance based on classification
df['is_relevant'] = df['classification'] == 'relevant'

# Remove irrelevant rows
df_cleaned = df[df['is_relevant']]

# Function to summarize text
def summarize_text(row):
    summary = summarizer(row['text'], max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Apply summarization
df_cleaned['summary'] = df_cleaned.apply(summarize_text, axis=1)

# Generate metadata
df_cleaned['metadata'] = df_cleaned.apply(lambda row: {
    'document': row['document'],
    'section': row['section']
}, axis=1)


### Embed and Store in Vector Database
from langchain.embeddings import OpenAIEmbeddings

# Initialize the embedding model
embedding_model = OpenAIEmbeddings()

# Function to embed text
def embed_text(row):
    embedding = embedding_model.embed_text(row['summary'])
    return embedding

# Apply embedding
df_cleaned['embedding'] = df_cleaned.apply(embed_text, axis=1)

# Example: Store embeddings in PGVector (not executable in this environment)
import psycopg2

def store_embeddings(df, conn):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT INTO your_table (text, embedding, metadata) VALUES (%s, %s, %s)",
            (row['summary'], row['embedding'].tolist(), row['metadata'])
        )
    conn.commit()

# Assuming you have a PostgreSQL connection
conn = psycopg2.connect(database="your_db", user="your_user", password="your_password", host="your_host", port="your_port")
store_embeddings(df_cleaned, conn)

