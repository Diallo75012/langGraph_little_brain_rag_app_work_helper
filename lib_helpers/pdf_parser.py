# PDF load and then parse the file to get what is structured and usable by llm for embedding/retrival but only the quality data, the rest will need to be searched online using some other tools

import pandas as pd
import PyPDF2
import re
import psycopg2
from uuid import uuid4
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
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

# Function to extract text and sections from a PDF
def pdf_to_sections(pdf_path) -> list:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    
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
        print("Line: ", line)
        line = line.strip()
        if not line:
            continue

        # Check for Titles, Headings, Subheadings, etc.
        if is_title(line):
            print("Line is title: ", line)
            if section_buffer:
                data.append({
                    'text': ' '.join(section_buffer),
                    'section': current_section,
                    'document': pdf_path
                })
                section_buffer = []
            current_section = line
        elif is_bullet_point(line):
            print("Is bullet point: ", line)
            section_buffer.append(f"Bullet Point: {line}")
        elif is_table_line(line):
            print("Is table: ", line)
            section_buffer.append(f"Table Row: {line}")
        else:
            print("Is just a line", line)
            section_buffer.append(line)
    
    # Add any remaining buffer to data
    if section_buffer:
        data.append({
            'text': ' '.join(section_buffer),
            'section': current_section,
            'document': pdf_path
        })

    print("Section Buffer: ", section_buffer)
    return data

