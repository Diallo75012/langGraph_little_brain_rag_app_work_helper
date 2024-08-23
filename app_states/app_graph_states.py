import os
from pydantic import BaseModel, validator
import pandas as pd
from typing import List, Dict, Any, Optional
from prompts.prompts import (
  detect_content_type_prompt,
  summarize_text_prompt,
  generate_title_prompt
)
from lib_helpers.embedding_and_retrieval import (
  COLLECTION_NAME,
  CONNECTION_STRING
)
from typing import Dict, List, Tuple, Any, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

class StateCustom(BaseModel):
  # user input
  user_initial_input: str = ""
  user_reformulated_query: str = ""
  # detect content
  doc_type: str = ""
  no_doc_but_simple_query_text: str = ""
  # process_query
  maximum_content_length: int = 200
  maximum_title_length: int = 30
  chunk_size_df: int = 250
  detect_prompt_template: dict = detect_content_type_prompt
  text_summary_prompt: dict = summarize_text_prompt
  title_summary_prompt: dict = generate_title_prompt
  # parsed document
  pdf_parsed_dataframe_state: Optional[dict] = None
  url_parsed_dataframe_state: Optional[dict] = None
  table_name: str = "test_table"
  # custom_chunk_and_embed_to_vectordb
  #table_name: str = "test_table"
  chunk_size_db: int = 500
  COLLECTION_NAME: str = COLLECTION_NAME
  CONNECTION_STRING: str = CONNECTION_STRING
  # query_redis_cache_then_vecotrdb_if_no_cache
  #table_name: str = "test_table"
  score: float = 0.3
  top_n: int = 2
  # func outputs
  df_final: Optional[dict] = None
  content_dict: dict = {}
  # stored df to database
  df_stored: bool = False
  # chunked and embedded chunk to db
  df_data_chunked_and_embedded: bool = False
  # retrieved redis cache then vector db
  query_hash_retrieved: list = []
  query_vector_hash_retrieved: list = []
  vectorbd_retrieved: list = []
  nothing_retrieved: bool = False # if True means nothing has been retrieved fallback to internet search

  @validator("pdf_parsed_dataframe_state", "url_parsed_dataframe_state", "df_final", pre=True, always=True)
  def check_dataframe_state(cls, v):
    if isinstance(v, pd.DataFrame):
      return v.to_dict()
    return v

  class Config:
    arbitrary_types_allowed = True











