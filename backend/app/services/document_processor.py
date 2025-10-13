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
import re
import os
import asyncio

class DocumentProcessor:
    """Process PDF documents and extract structured data"""
    
    def __init__(self, db: Session):
        self.table_parser = TableParser()
        self.vector_store = VectorStore(db=db) 
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
                pdf = pdfplumber.open(file_path)
            except Exception as e:
                raise IOError(f"Failed to open or read PDF file: {e}")

            with pdf:
                # Basic check for empty PDF
                if not pdf.pages:
                    raise ValueError("PDF opened successfully but contains no pages.")
                    
                # Extract Fund Data (from first page)
                first_page_text = pdf.pages[0].extract_text()
                fund_data = self._extract_fund_data_from_text(first_page_text)
                
                if not fund_data.get("name") or not fund_data.get("gp_name"):
                    # NOTE: Simplified error handling
                    fund = Fund(name=f"Unknown Fund {fund_id}", gp_name="Unknown GP", vintage_year=2024)
                    print("Warning: Could not extract full Fund data. Using placeholders.")
                else:
                    fund = self._get_or_create_fund(fund_data)

                document.fund_id = fund.id # Link document
                processing_stats["fund_id"] = fund.id
                
                # --- 2. Table Extraction ---
                all_raw_tables = []
                raw_text_pages = {}
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    
                    # Table extraction
                    raw_tables = page.extract_tables(table_settings={}) 
                    all_raw_tables.extend([
                        {"page": page_num, "raw_data": t} for t in raw_tables
                    ])
                    
                    # Text extraction
                    raw_text_pages[page_num] = page.extract_text() or ""

            # 3. Classify and Parse Tables
            parsed_data = self._classify_and_parse_tables(all_raw_tables)
            
            # 4. Store Transactions
            processing_stats["calls_stored"] = self._store_capital_calls(fund.id, parsed_data.get("capital_calls", []))
            processing_stats["distributions_stored"] = self._store_distributions(fund.id, parsed_data.get("distributions", []))
            processing_stats["adjustments_stored"] = self._store_adjustments(fund.id, parsed_data.get("adjustments", []))

            # 5. Text Chunking (Unstructured Data Path)
            text_chunks = self._chunk_text(document.id, raw_text_pages)
            
            # 6. Store Chunks in Vector Database (Vector Storage)
            chunks_stored = await self._store_chunks(document.id, fund.id, text_chunks)
            processing_stats["chunks_stored"] = chunks_stored
            
            # Final status update
            self.db.commit() # Commit all changes from this function
            processing_stats["status"] = "completed"
            processing_stats["message"] = "Document processed, transactions and vectors stored."
            return processing_stats
            
        except (IOError, ValueError, FileNotFoundError, Exception) as e:
            # Catch all known exceptions, including the new FileNotFoundError
            error_message = str(e)
            print(f"FATAL PROCESSING ERROR for document {document.id}: {error_message}")
            self.db.rollback()
            processing_stats["error"] = error_message
            return processing_stats
            
    # --- Helper Methods for Fund Extraction (Unchanged from previous response) ---
    def _extract_fund_data_from_text(self, text: str) -> Dict[str, Any]:
        """Utility to extract key fund data from the report text."""
        data = {}
        match = re.search(r"Fund Name:\s*([^\n]+)", text)
        data["name"] = match.group(1).strip() if match else None
        match = re.search(r"GP:\s*([^\n]+)", text)
        data["gp_name"] = match.group(1).strip() if match else None
        match = re.search(r"Vintage Year:\s*(\d{4})", text)
        data["vintage_year"] = int(match.group(1)) if match else None
        match = re.search(r"Fund Size:\s*(\$[^ \n]+)", text)
        if match:
             data["fund_size_str"] = match.group(1).strip().replace("$", "").replace(",", "")
        match = re.search(r"Report Date:\s*([^\n]+)", text)
        data["report_date_str"] = match.group(1).strip() if match else None
        return data

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

    # --- Table Classification and Storage Methods ---

    def _classify_and_parse_tables(self, raw_tables: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Uses the TableParser to classify and structure raw table data."""
        
        parsed_data = {
            "capital_calls": [],
            "distributions": [],
            "adjustments": []
        }
        
        for raw_table_obj in raw_tables:
            table_data = raw_table_obj["raw_data"]
            page_num = raw_table_obj["page"]
            
            try:
                table_type, transactions = self.table_parser.parse_table(
                    table_data, 
                    page_number=page_num
                )
                
                if table_type and table_type in parsed_data:
                    parsed_data[table_type].extend(transactions)
                    
            except Exception as e:
                # Log the error for a specific table
                print(f"Error parsing table on page {page_num}: {e}")
                
        return parsed_data

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
    
    def _chunk_text(self, document_id: int, raw_text_pages: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Chunks raw text into small pieces suitable for embedding, using a simple
        paragraph-splitting strategy while respecting a maximum chunk size.
        
        Args:
            document_id: ID of the source document.
            raw_text_pages: Dictionary mapping page number (int) to raw text (str).
            
        Returns:
            List of text chunks with metadata (content, metadata: {document_id, page}).
        """
        chunks = []
        max_chunk_size = 500 # Define a reasonable max size for embeddings
        
        for page_num, text_content in raw_text_pages.items():
            # 1. Split text into paragraphs based on one or more newlines
            paragraphs = re.split(r'\n\s*\n', text_content)
            current_chunk = ""
            
            for p in paragraphs:
                p = p.strip()
                if not p: 
                    continue

                # Check if adding the current paragraph exceeds the max size
                # We account for 2 characters for the separator ("\n\n")
                if len(current_chunk) + len(p) + 2 <= max_chunk_size:
                    current_chunk += p + "\n\n"
                else:
                    # If the current chunk is not empty, save it
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk.strip(),
                            # Add basic metadata including the page number
                            "metadata": {"document_id": document_id, "page": page_num}
                        })
                    # Start a new chunk with the current paragraph
                    current_chunk = p + "\n\n"
            
            # Save any remaining content in the current chunk after the page loop finishes
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {"document_id": document_id, "page": page_num}
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
                **chunk["metadata"] # includes page number
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
