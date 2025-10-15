"""
Vector store service using pgvector (PostgreSQL extension)
"""
from typing import List, Dict, Any, Optional
import numpy as np
import json
import asyncio
import aiohttp
from sqlalchemy.orm import Session
from sqlalchemy import text
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings
from app.db.session import SessionLocal
from app.schemas.vector_store import DocumentChunk


class VectorStore:
    """pgvector-based vector store for document embeddings"""
    
    def __init__(self, db: Session = None):
        self.db = db or SessionLocal()
        self.embeddings = self._initialize_embeddings()
        self._ensure_extension()
    
    def _initialize_embeddings(self):
        """Initialize embedding model based on settings"""
        # NOTE: This method remains unchanged to preserve your intended logic 
        # for selecting the embedding model based on the settings file.
        if settings.GEMINI_API_KEY == "": 
            # Use Gemini API if configured correctly for the runtime environment
            return {
                "type": "gemini",
                "model_name": settings.GEMINI_EMBEDDING_MODEL,
                "api_key": settings.GEMINI_API_KEY, # This is the placeholder ""
                "dimension": 768 
            }
        else:
            # Fallback to local embeddings (e.g., if running outside Canvas or key is explicitly provided)
            print("Falling back to local HuggingFace embeddings.")
            return {
                "type": "huggingface",
                "model_name": settings.FALLBACK_EMBEDDING_MODEL_NAME,
                "instance": HuggingFaceEmbeddings(
                    model_name=settings.FALLBACK_EMBEDDING_MODEL_NAME
                ),
                "dimension": 384 
            }
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled and the embeddings table exists.
        """
        try:
            # Enable pgvector extension
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Use dimension from initialized config
            dimension = self.embeddings["dimension"]
            
            # Create embeddings table
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    fund_id INTEGER,
                    content TEXT NOT NULL,
                    embedding vector({dimension}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Index for efficient similarity search
                -- Using lists=100 is a reasonable default for IVFFlat
                CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
                ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            
            self.db.execute(text(create_table_sql))
            self.db.commit()
        except Exception as e:
            print(f"Error ensuring pgvector extension: {e}")
            self.db.rollback()
    
    async def _get_embedding_async(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using the configured model (Gemini or HuggingFace).
        This implements a TEMPORARY direct API key injection for easy development.
        """
        config = self.embeddings
        
        if config["type"] == "gemini":
            # --- TEMPORARY DIRECT API KEY INJECTION FOR DEVELOPMENT ---
            
            # 1. PASTE YOUR KEY HERE for testing. REMOVE IT before deployment.
            TEMP_DEV_API_KEY = settings.GOOGLE_API_KEY 
            
            if not TEMP_DEV_API_KEY:
                print("Error: GOOGLE_API_KEY is missing for Gemini embedding")
                raise RuntimeError("Missing Gemini API Key.")

            # 2. Use the key as a query parameter (the simple method)
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config['model_name']}:embedContent?key={TEMP_DEV_API_KEY}"
            
            # 3. Only need basic headers now
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "content": {"parts": [{"text": text}]}
            }
            
            MAX_RETRIES = 5
            initial_delay = 1.0
            
            for attempt in range(MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession(headers=headers) as session: 
                        async with session.post(api_url, json=payload) as response:
                            if response.status == 200:
                                result = await response.json()
                                # Extract embedding values
                                embedding_list = result.get('embedding', {}).get('values')
                                if embedding_list:
                                    return np.array(embedding_list, dtype=np.float32)
                                else:
                                    raise ValueError("Gemini API returned no embedding values.")
                            
                            # Handle retryable errors (Rate Limit, Server Errors)
                            elif response.status in [429, 500, 503]:
                                print(f"Gemini Embedding retryable error: Status {response.status}. Retrying...")
                                raise ConnectionError("Retryable API error")
                            else:
                                # Handle non-retryable errors (like 400 INVALID_ARGUMENT/403 PERMISSION_DENIED)
                                error_details = await response.json()
                                print(f"Gemini Embedding non-retryable error (Status {response.status}): {error_details}")
                                error_message = error_details.get('error', {}).get('message', 'Unknown Error')
                                raise ValueError(f"Gemini embedding API failed: {error_message}")

                except ConnectionError:
                    if attempt < MAX_RETRIES - 1:
                        delay = initial_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        raise RuntimeError("Failed to get embedding after multiple retries.")
                except ValueError as e:
                    raise e
                except Exception as e:
                    print(f"Error during Gemini embedding call: {e}")
                    raise RuntimeError(f"Embedding generation failed: {e}")

            raise RuntimeError("Gemini embedding generation failed.")
            # --- End TEMPORARY Implementation ---

        elif config["type"] == "huggingface":
            # Fallback for local HuggingFace embeddings
            loop = asyncio.get_event_loop()
            embedding_list = await loop.run_in_executor(
                None, 
                lambda: config["instance"].embed_documents([text])[0]
            )
            return np.array(embedding_list, dtype=np.float32)

        else:
            raise NotImplementedError("Unknown embedding type.")
    
    async def add_document(self, content: str, metadata: Dict[str, Any]):
        """
        Add a document to the vector store.
        """
        try:
            # Generate embedding
            embedding = await self._get_embedding_async(content)
            embedding_list = embedding.tolist() 
            
            # Insert into database
            insert_sql = text("""
                INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                VALUES (:document_id, :fund_id, :content, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
            """)
            
            self.db.execute(insert_sql, {
                "document_id": metadata.get("document_id"),
                "fund_id": metadata.get("fund_id"),
                "content": content,
                "embedding": str(embedding_list),
                "metadata": json.dumps(metadata)
            })
            self.db.commit()
        except Exception as e:
            print(f"Error adding document: {e}")
            self.db.rollback()
            raise

    async def similarity_search(self, query: str, k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the vector store for the top k most similar documents to the query.
        """
        try:
            # Generate embedding for the query
            query_embedding = await self._get_embedding_async(query)
            query_embedding_list = query_embedding.tolist()

            # --- FIX: Use CAST() instead of ::vector to avoid SyntaxError ---
            sql_query = """
                SELECT 
                    id,
                    document_id,
                    fund_id,
                    content,
                    metadata,
                    1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity_score
                FROM document_embeddings
            """
            
            params = {
                "query_embedding": str(query_embedding_list)
            }
            
            # Build WHERE clause for filtering (e.g., by fund_id)
            where_clauses = []
            if filter_metadata:
                if "fund_id" in filter_metadata:
                    where_clauses.append("fund_id = :fund_id")
                    params["fund_id"] = filter_metadata["fund_id"]

            if where_clauses:
                sql_query += " WHERE " + " AND ".join(where_clauses)

            # Order by vector distance (closest first)
            sql_query += " ORDER BY embedding <=> CAST(:query_embedding AS vector)"
            
            # Add LIMIT for the top k results
            sql_query += " LIMIT :k"
            params["k"] = k

            # Execute the query
            result = self.db.execute(text(sql_query), params)
            
            # Format results
            chunks = []
            for row in result:
                chunks.append(DocumentChunk(
                    id=row[0],
                    document_id=row[1],
                    fund_id=row[2],
                    content=row[3],
                    metadata=row[4],
                    score=row[5]
                ).model_dump())
            
            return chunks

        except Exception as e:
            # We must print the specific error for debugging
            print(f"Error in similarity search: {e}")
            raise

    def clear(self, fund_id: Optional[int] = None):
        """
        Clear the vector store, optionally filtered by fund_id.
        """
        try:
            if fund_id:
                delete_sql = text("DELETE FROM document_embeddings WHERE fund_id = :fund_id")
                self.db.execute(delete_sql, {"fund_id": fund_id})
            else:
                delete_sql = text("DELETE FROM document_embeddings")
                self.db.execute(delete_sql)
            
            self.db.commit()
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            self.db.rollback()