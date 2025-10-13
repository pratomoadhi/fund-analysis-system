from typing import Dict, List, Any, Tuple
import re
from datetime import datetime

class TableParser:
    """
    Analyzes raw tables extracted by pdfplumber, classifies them (Capital Calls, 
    Distributions, Adjustments), and cleans/normalizes the transaction data.
    """

    def __init__(self):
        # Define the set of required headers for each transaction type for classification
        self.classification_schemas = {
            "capital_calls": {"Date", "Call Number", "Amount", "Description"},
            "distributions": {"Date", "Type", "Amount", "Recallable", "Description"},
            "adjustments": {"Date", "Type", "Amount", "Description"},
        }
        # Common date formats to try when parsing
        self.date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y"]
        
    def _clean_header(self, raw_header: str) -> str:
        """Standardizes header strings by stripping whitespace and newlines."""
        if raw_header is None:
            return ""
        # Replace newlines/tabs, strip leading/trailing space, and replace multiple spaces with single space
        return re.sub(r'\s+', ' ', raw_header.strip().replace('\n', ' ').strip())
        
    def _clean_amount(self, raw_amount: str) -> float:
        """Removes currency symbols and commas, converting the result to a float."""
        if not raw_amount:
            return 0.0
        
        cleaned = raw_amount.strip().replace('\n', '')
        
        is_negative = False
        # 1. Check for financial negative notation (e.g., (1,000) or -$1,000)
        if cleaned.startswith('-'):
            is_negative = True
            cleaned = cleaned[1:]
        elif '(' in cleaned and ')' in cleaned: 
            is_negative = True
            cleaned = cleaned.replace('(', '').replace(')', '')
        
        # 2. Remove common clutter
        cleaned = cleaned.replace('$', '').replace(',', '').strip()
        
        if not cleaned:
             return 0.0

        try:
            value = float(cleaned)
            return -value if is_negative else value
        except ValueError:
            # Fallback for unexpected format
            print(f"Warning: Failed to convert amount string '{raw_amount}' to float. Using 0.0.")
            return 0.0

    def _clean_date(self, raw_date: str) -> str:
        """Converts raw date string to YYYY-MM-DD format for storage, trying multiple formats."""
        if not raw_date:
            raise ValueError("Date field is empty.")
            
        date_str = raw_date.strip()
        
        for fmt in self.date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        # If no format matched, raise a definitive error
        raise ValueError(f"Date string '{date_str}' does not match any expected format.")


    def _classify_table(self, headers: List[str]) -> str | None:
        """Determines the type of transaction table based on its clean headers."""
        
        # Create a set of combined, cleaned header keywords for robust comparison
        clean_headers_set = set()
        for h in headers:
            # We look for key words without spaces to make classification less sensitive to formatting
            clean_headers_set.add(h.replace(" ", "").replace("_", "").lower())
        
        # Define keywords for comparison
        capital_call_keywords = {"date", "callnumber", "amount", "description"}
        distribution_keywords = {"date", "type", "amount", "recallable", "description"}
        adjustment_keywords = {"date", "type", "amount", "description"}
        
        # 1. Distributions check (most specific)
        if clean_headers_set.issuperset(distribution_keywords):
            return "distributions"
        
        # 2. Capital Calls check
        if clean_headers_set.issuperset(capital_call_keywords):
            return "capital_calls"

        # 3. Adjustments check
        # Check if it has the core adjustment keywords but is not a distribution/call
        if clean_headers_set.issuperset(adjustment_keywords) and not clean_headers_set.intersection({"callnumber", "recallable"}):
            return "adjustments"
            
        return None # Unclassified

    
    def parse_table(self, raw_data: List[List[str]], page_number: int = 1) -> Tuple[str | None, List[Dict[str, Any]]]:
        """
        Main method to classify a raw table and normalize its data.
        
        ... (docstring omitted for brevity)
        """
        if not raw_data or len(raw_data) < 2:
            return None, []

        raw_headers = raw_data[0]
        # Clean headers to use them as keys in the row map
        headers = [self._clean_header(h) for h in raw_headers]
        table_type = self._classify_table(headers)
        
        if not table_type:
            return None, []
            
        rows_data = raw_data[1:]
        transactions = []
        
        for row in rows_data:
            # Basic validation: Skip rows where most cells are empty or None
            non_empty_cells = [cell for cell in row if cell and cell.strip()]
            if len(non_empty_cells) < 2: 
                continue # Skip if not enough data to be a valid transaction
                
            try:
                if table_type == "capital_calls":
                    transactions.append(self._parse_capital_call_row(headers, row))
                elif table_type == "distributions":
                    transactions.append(self._parse_distribution_row(headers, row))
                elif table_type == "adjustments":
                    transactions.append(self._parse_adjustment_row(headers, row))
                    
            except ValueError as ve:
                # Catching specific errors from cleaning (like bad date)
                print(f"Validation Error processing row on page {page_number} for type {table_type}: {ve}")
                continue
            except Exception as e:
                print(f"Unexpected Error processing row on page {page_number} for type {table_type}: {e}")
                continue
                
        return table_type, transactions

    # --- Row Parsing Helper Methods (Slightly adjusted for robustness) ---

    def _parse_capital_call_row(self, headers: List[str], row: List[str]) -> Dict[str, Any]:
        """Parses and normalizes a single Capital Call row."""
        row_map = dict(zip(headers, row))
        
        # Required field validation
        if not row_map.get("Date") or not row_map.get("Amount"):
            raise ValueError("Missing required Date or Amount for Capital Call.")

        return {
            "date": self._clean_date(row_map.get("Date")),
            "call_type": row_map.get("Call Number", "").strip(),
            "amount": self._clean_amount(row_map.get("Amount")),
            "description": row_map.get("Description", "").strip().replace('\n', ' ')
        }

    def _parse_distribution_row(self, headers: List[str], row: List[str]) -> Dict[str, Any]:
        """Parses and normalizes a single Distribution row."""
        row_map = dict(zip(headers, row))
        
        # Required field validation
        if not row_map.get("Date") or not row_map.get("Amount"):
            raise ValueError("Missing required Date or Amount for Distribution.")
            
        recallable_str = row_map.get("Recallable", "").strip().lower()
        
        return {
            "date": self._clean_date(row_map.get("Date")),
            "type": row_map.get("Type", "").strip(), 
            "amount": self._clean_amount(row_map.get("Amount")),
            "recallable": recallable_str in ("yes", "true", "y"),
            "description": row_map.get("Description", "").strip().replace('\n', ' ')
        }

    def _parse_adjustment_row(self, headers: List[str], row: List[str]) -> Dict[str, Any]:
        """Parses and normalizes a single Adjustment row."""
        row_map = dict(zip(headers, row))
        
        # Required field validation
        if not row_map.get("Date") or not row_map.get("Amount"):
            raise ValueError("Missing required Date or Amount for Adjustment.")
            
        adj_type = row_map.get("Type", "").strip()
        
        is_contribution_adjustment = "contribution" in adj_type.lower()
        category = "Other"
        if "recallable" in adj_type.lower():
            category = "Distribution Recall"
        elif "call" in adj_type.lower():
            category = "Capital Call"
            
        return {
            "date": self._clean_date(row_map.get("Date")),
            "type": adj_type,
            "amount": self._clean_amount(row_map.get("Amount")),
            "is_contribution_adjustment": is_contribution_adjustment,
            "category": category,
            "description": row_map.get("Description", "").strip().replace('\n', ' ')
        }
