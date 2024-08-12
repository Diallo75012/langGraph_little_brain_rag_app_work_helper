# PDF load and then parse the file to get what is structured and usable by llm for embedding/retrival but only the quality data, the rest will need to be searched online using some other tools

import pandas as pd
import PyPDF2
import re
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
    prompt_text = f"Please summarize the following text in no more than {maximum} characters:\n\n{row['text']}. Answer only with the schema in markdown between ```python ```"
    
    response = llm.invoke(prompt_text)
    summary = response.content.split("```")[1].strip("python").strip()
    #summary = response['choices'][0]['text'].strip()
    print("Summary: ", summary)
    return summary

def generate_title(llm, row, maximum):
    prompt_text = f"Please create a title in no more than {maximum} characters for:\n\n{row['section']} - {row['text']}.\nAnswer only with the schema in markdown between ```python ```"
    response = llm.invoke(prompt_text)
    title = response.content.split("```")[1].strip("python").strip()
    return title

def get_final_df(llm: ChatGroq, pdf_file_name_path: str, maximum_content_length: int, maximum_title_length: int):
  # Parse the PDF and create a DataFrame
  pdf_data = pdf_to_sections(pdf_file_name_path)
  print("PDF DATA: ", pdf_data)
  df = pd.DataFrame(pdf_data)
  print("DF: ", df)

  # Apply summarization
  df['summary'] = df.apply(lambda row: summarize_text(llm, row, maximum_content_length), axis=1) # here use llm to make the summary

  # Generate titles using section and text
  df['title'] = df.apply(lambda row: generate_title(llm, row, maximum_title_length), axis=1)

  # Generate metadata and add UUID and retrieved fields
  df['id'] = [str(uuid4()) for _ in range(len(df))]
  df['doc_name'] = df['document'].apply(lambda x: os.path.basename(x)) # or just use function var `pdf_file_name`
  df['content'] = df['summary']
  df['retrieved'] = False

  # Select only the necessary columns
  df_final = df[['id', 'doc_name', 'title', 'content', 'retrieved']]
  print("Df Final from library: ", df_final, type(df_final))
  with open("test_pdf_parser_df_output.csv", "w", encoding="utf-8") as f:
    df_final.to_csv(f, index=False)
  return df_final

