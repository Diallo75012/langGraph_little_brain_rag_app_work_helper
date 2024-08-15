from pydantic import BaseModel


class ParseDocuments(BaseModel):
    webpage_parsed_dataframe_state: pd.DataFrame
    url_parsed_dataframe_state: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True
