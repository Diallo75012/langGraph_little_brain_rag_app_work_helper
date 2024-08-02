"""
Here we will put all function that will be used to analyze, decompose, rephrase or other type of operation on user initial query
"""
import re
from typing import Tuple
# Utility function to determine if the query contains a PDF or URL
def detect_content_type(query: str) -> Tuple:
    """
    Utility function to determine if the query contains a PDF or URL.

    Args:
    query (str): The user query string.

    Returns:
    tuple: A tuple containing the type of content ('pdf', 'url', or 'text') and the detected content (URL or query).
    """
    pdf_pattern = r"https?://[^\s]+\.pdf"
    url_pattern = r"https?://[^\s]+"

    if re.search(pdf_pattern, query):
        return 'pdf', re.search(pdf_pattern, query).group(0)
    elif re.search(url_pattern, query):
        return 'url', re.search(url_pattern, query).group(0)
    else:
        return 'text', query
