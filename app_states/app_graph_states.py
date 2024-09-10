from pydantic import BaseModel, validator
from typing import Dict, List, Tuple, Any, Optional


class GraphStatePersistFlow(BaseModel):
  user_initial_query: str = ""
  query_reformulated: str = ""
  primary_graph_last_message: str = ""
  subgraph_messages: List[str] = []
  parquet_file_path: str = ""











