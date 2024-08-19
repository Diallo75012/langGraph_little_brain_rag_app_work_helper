from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any, Optional

class ParseDocuments(BaseModel):
    webpage_parsed_dataframe_state: pd.DataFrame
    url_parsed_dataframe_state: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

class TestState(BaseModel):
    int_state: int
    array_state: List
    dict_state: Dict
