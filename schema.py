# schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AnalysisOutput(BaseModel):
    """Output schema for the AI Data Analyst's response."""
    answer: str = Field(description="A concise answer or insight based on the user's query.")
    query_sql: Optional[str] = Field(None, description="The pandas query string used if data was queried or filtered.")
    data_preview: Optional[List[Dict[str, Any]]] = Field(None, description="A preview of the data, if a query was performed and results are concise (e.g., first 5 rows).")
    chart_path: Optional[str] = Field(None, description="The file path to a generated chart image, if a plot was requested.")