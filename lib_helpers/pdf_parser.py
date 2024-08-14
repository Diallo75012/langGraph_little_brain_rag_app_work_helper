# PDF load and then parse the file to get what is structured and usable by llm for embedding/retrival but only the quality data, the rest will need to be searched online using some other tools

import pandas as pd
import PyPDF2
import re
from dotenv import load_dotenv
import os
from typing import Dict, Any, List, Optional


load_dotenv()

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

