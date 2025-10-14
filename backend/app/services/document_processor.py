"""
Document processing service using pdfplumber

TODO: Implement the document processing pipeline
- Extract tables from PDF using pdfplumber
- Classify tables (capital calls, distributions, adjustments)
- Extract and chunk text for vector storage
- Handle errors and edge cases
"""
from typing import Dict, List, Any
from datetime import datetime
import pdfplumber
from app.services.table_parser import TableParser
from app.services.vector_store import VectorStore
from app.models.document import Document
from app.models.fund import Fund
from app.models.transaction import CapitalCall, Distribution, Adjustment
from sqlalchemy.orm import Session
from docling.document_converter import DocumentConverter
import os
import asyncio

class DocumentProcessor:
    """Process PDF documents and extract structured data"""
    
    def __init__(self, db: Session):
        self.table_parser = TableParser()
        self.vector_store = VectorStore(db=db) 
        self.converter = DocumentConverter()
        self.db = db
    
    async def process_document(self, file_path: str, document_id: int, fund_id: int) -> Dict[str, Any]:
        """
        Process a PDF document
        
        TODO: Implement this method
        - Open PDF with pdfplumber
        - Extract tables from each page
        - Parse and classify tables using TableParser
        - Extract text and create chunks
        - Store chunks in vector database
        - Return processing statistics
        
        Args:
            file_path: Path to the PDF file
            document_id: Database document ID
            fund_id: Fund ID
            
        Returns:
            Processing result with statistics
        """
        # TODO: Implement PDF processing logic
        document = self.db.query(Document).filter(Document.id == document_id).first()
        
        processing_stats = {
            "status": "failed", 
            "fund_id": None,
            "calls_stored": 0,
            "distributions_stored": 0,
            "adjustments_stored": 0,
            "chunks_stored": 0,
            "error": None
        }
        
        pdf = None # Initialize pdf outside the block
        fund = None
        try:
            # --- Robust File Existence Check ---
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at path: {file_path}")
            
            # --- 1. Robustly Open PDF and Extract Fund Data ---
            try:
                # Catch specific errors related to file access or corruption during open
                pdf = self.converter.convert(file_path)
            except Exception as e:
                raise IOError(f"Failed to open or read PDF file: {e}")
            
            # We rely on the result containing the document as a dictionary structure
            if pdf and hasattr(pdf, 'document') and pdf.document:
                
                # 1. Check if the document is already a dict
                if isinstance(pdf.document, dict):
                    doc = pdf.document
                
                # 2. Prioritize export_to_dict() based on the user's successful method
                if hasattr(pdf.document, 'export_to_dict') and callable(pdf.document.export_to_dict):
                    doc = pdf.document.export_to_dict()
    
                # Parse texts
                if 'texts' not in doc:
                    raise ValueError("'texts' section not found. Cannot chunk by all texts.")

                text_container = doc['texts']
                
                # Determine how to iterate based on type
                if isinstance(text_container, dict):
                    # If it's a dictionary, iterate over the values (the actual text node objects)
                    text_contents = text_container.values()
                elif isinstance(text_container, list):
                    # If it's a list, iterate over the elements (the actual text node objects)
                    text_contents = text_container
                else:
                    raise ValueError("'texts' section exists but is neither a dictionary nor a list. Cannot chunk by all texts.")
                
                text_chunks = self._chunk_text(text_contents)     
                text_chunks_dict = {item["id"]: item for item in text_chunks}   

                # Extract Fund Data
                if 'groups' in doc and doc['groups']:
                    fund_group = doc['groups'][0]
                    fund_data = self._extract_fund_data_from_text(text_chunks_dict, fund_group)
                
                    if fund_data.get("name") and fund_data.get("gp_name"):
                        fund = self._get_or_create_fund(fund_data)
                        fund_id = fund.id

                # Extract tables
                if 'tables' in doc and isinstance(doc['tables'], list):
                    # Parse tables
                    tables = self.table_parser.extract_tables(doc['tables'], doc['body']['children'], text_chunks_dict)

                    # Store tables
                    processing_stats["calls_stored"] = self._store_capital_calls(fund_id, tables.get("capital_calls", {}).get("rows", []))
                    processing_stats["distributions_stored"] = self._store_distributions(fund_id, tables.get("distributions", {}).get("rows", []))
                    processing_stats["adjustments_stored"] = self._store_adjustments(fund_id, tables.get("adjustments", {}).get("rows", []))

                document.fund_id = fund_id # Link document
                processing_stats["fund_id"] = fund_id
                
                # Store chunks            
                chunks_stored = await self._store_chunks(document.id, fund_id, text_chunks)
                processing_stats["chunks_stored"] = chunks_stored     

            else:
                raise ValueError("Conversion succeeded, but the resulting document object was empty.")
            
            # Final status update
            self.db.commit() # Commit all changes from this function
            processing_stats["status"] = "completed"
            processing_stats["message"] = "Document processed, transactions and vectors stored."
            print(processing_stats)
            return processing_stats
            
        except (IOError, ValueError, FileNotFoundError, Exception) as e:
            # Catch all known exceptions, including the new FileNotFoundError
            error_message = str(e)
            print(f"FATAL PROCESSING ERROR for document {document.id}: {error_message}")
            self.db.rollback()
            processing_stats["error"] = error_message
            return processing_stats
            
    # --- Helper Methods for Fund Extraction (Unchanged from previous response) ---

    def _extract_fund_data_from_text(self, text_chunks_dict: Dict[str, Any], fund_group: Dict[str, Any]) -> Dict[str, Any]:
        """Utility to extract key fund data from the report text."""

        transformed = {} 
        raw_data = []
        if 'children' in fund_group and fund_group['children']:
            for item in fund_group['children']:
                node_id = item["$ref"].split('/')[-1]
                raw_data.append(text_chunks_dict.get(node_id, {"content": ""}))

        if raw_data:
            raw_fund_data = {
                raw_data[i]["content"].rstrip(":"): raw_data[i + 1]["content"]
                for i in range(0, len(raw_data) - 1, 2)
            }

            transformed = {
                "name": raw_fund_data.get("Fund Name"),
                "gp_name": raw_fund_data.get("GP"),
                "fund_type": raw_fund_data.get("Fund Type"),
                "vintage_year": int(raw_fund_data.get("Vintage Year") or 2025)
            }

        return transformed

    def _get_or_create_fund(self, fund_data: Dict[str, Any]) -> Fund:
        """Checks for existing fund or creates a new one."""
        existing_fund = self.db.query(Fund).filter(
            Fund.name == fund_data["name"],
            Fund.vintage_year == fund_data["vintage_year"]
        ).first()
        
        if existing_fund:
            return existing_fund
            
        new_fund = Fund(
            name=fund_data["name"],
            gp_name=fund_data.get("gp_name"),
            vintage_year=fund_data.get("vintage_year"),
        )
        self.db.add(new_fund)
        self.db.commit() 
        self.db.refresh(new_fund)
        return new_fund

    # --- Table Storage Methods ---

    def _store_capital_calls(self, fund_id: int, calls_data: List[Dict[str, Any]]) -> int:
        """Stores capital call records in the database."""
        new_calls = []
        for call in calls_data:
            new_calls.append(CapitalCall(
                fund_id=fund_id,
                call_date=datetime.strptime(call["date"], "%Y-%m-%d").date(),
                call_type=call["call_type"],
                amount=call["amount"],
                description=call["description"]
            ))
        
        self.db.add_all(new_calls)
        return len(new_calls)

    def _store_distributions(self, fund_id: int, distributions_data: List[Dict[str, Any]]) -> int:
        """Stores distribution records in the database."""
        new_distributions = []
        for dist in distributions_data:
            new_distributions.append(Distribution(
                fund_id=fund_id,
                distribution_date=datetime.strptime(dist["date"], "%Y-%m-%d").date(),
                distribution_type=dist["type"],
                is_recallable=dist["recallable"],
                amount=dist["amount"],
                description=dist["description"]
            ))
        
        self.db.add_all(new_distributions)
        return len(new_distributions)

    def _store_adjustments(self, fund_id: int, adjustments_data: List[Dict[str, Any]]) -> int:
        """Stores adjustment records in the database."""
        new_adjustments = []
        for adj in adjustments_data:
            new_adjustments.append(Adjustment(
                fund_id=fund_id,
                adjustment_date=datetime.strptime(adj["date"], "%Y-%m-%d").date(),
                adjustment_type=adj["type"],
                category=adj["category"],
                amount=adj["amount"],
                is_contribution_adjustment=adj["is_contribution_adjustment"],
                description=adj["description"]
            ))
        
        self.db.add_all(new_adjustments)
        return len(new_adjustments)
        
    # --- Text Chunking and Storage Methods ---

    def _chunk_text(self, text_contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk text content for vector storage
        
        TODO: Implement intelligent text chunking
        - Split text into semantic chunks
        - Maintain context overlap
        - Preserve sentence boundaries
        - Add metadata to each chunk
        
        Args:
            text_content: List of text content with metadata
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []

        for node in text_contents:
            text_content = node.get('text', '').strip()

            if text_content:
                # Determine the type, falling back to 'text' if not specified
                node_label = node.get('label', 'text')
                
                # Use 'self_ref' for ID if available, otherwise use a counter
                node_id = node.get('self_ref', f"unknown_{len(chunks)}").split('/')[-1]
                
                chunks.append({
                    "content": text_content,
                    "id": node_id,
                    "type": node_label
                })

        return chunks

    async def _store_chunks(self, document_id: int, fund_id: int, text_chunks: List[Dict[str, Any]]) -> int:
        """
        Stores text chunks and their embeddings in the pgvector store via the VectorStore service.
        """
        chunks_stored = 0
        
        # Create a list of async tasks for concurrent embedding and storage
        tasks = []
        for chunk in text_chunks:
            metadata = {
                "document_id": document_id,
                "fund_id": fund_id,
                "chunk_id": chunk["id"],
                "chunk_type": chunk["type"]
            }
            # Schedule the add_document call (which calls the Gemini API)
            tasks.append(self.vector_store.add_document(chunk["content"], metadata))

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                # Log or handle errors from individual chunk storage gracefully
                print(f"Error storing chunk for document {document_id}: {result}")
            else:
                chunks_stored += 1
        
        return chunks_stored