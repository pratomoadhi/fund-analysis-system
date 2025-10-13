from pydantic import BaseModel
from typing import Dict, Any

class DocumentChunk(BaseModel):
    """
    Schema for a single chunk of a document retrieved from the vector store.
    This structure is used to pass retrieved context to the QueryEngine.
    """
    id: int
    document_id: int
    fund_id: int
    content: str
    metadata: Dict[str, Any]
    score: float
