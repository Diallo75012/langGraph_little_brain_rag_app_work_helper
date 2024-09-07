import os
from pydantic import BaseModel, validator
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv


load_dotenv()

class GraphStatePersistFlow(BaseModel):
  query_reformulated: str = ""
  primary_graph_last_message: str = ""
  subgraph_last_message: str = ""











