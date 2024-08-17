from pydantic import BaseModel
import pandas as pd


class ParseDocuments(BaseModel):
    webpage_parsed_dataframe_state: pd.DataFrame
    url_parsed_dataframe_state: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True
