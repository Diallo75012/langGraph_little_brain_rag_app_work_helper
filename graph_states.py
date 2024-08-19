from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any, Optional

class DetectContentState(BaseModel):
    question: str
    doc_type: str
    no_doc_but_simple_query_text: str

class ParseDocuments(BaseModel):
    pdf_parsed_dataframe_state: pd.DataFrame
    url_parsed_dataframe_state: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

class StoreDftodbState(BaseModel):
    df_stored: bool = False

class ChunkAndEmbedDbdataState(BaseModel):
    df_data_chunked_and_embedded: bool = False

class RetrievedRedisVectordb(BaseModel):
    query_hash_retrieved: List[Dict[str,Any]]
    query_vector_hash_retrieved: List[Dict[str,Any]]
    vectorbd_retrieved: List[Dict[str,Any]]
    nothing_retrieved: bool = False # if True means nothing has been retrieved fallback to internet search
