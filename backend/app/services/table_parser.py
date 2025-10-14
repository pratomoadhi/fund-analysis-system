from typing import Dict, List, Any
from datetime import datetime

class TableParser:
    """
    Analyzes raw tables extracted by pdfplumber, classifies them (Capital Calls, 
    Distributions, Adjustments), and cleans/normalizes the transaction data.
    """

    def __init__(self):
        # Common date formats to try when parsing
        self.date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y"]
        
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
    
    def extract_tables(self, table_container: List[Dict[str, Any]], ref_container: List[Dict[str, Any]], text_chunks_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to extract raw tables.
        
        ... (docstring omitted for brevity)
        """
        tables = {}
        table_refs = {}
        for idx, ref in enumerate(ref_container):
            if 'tables' in ref["$ref"]:
                table_id = ref.get('$ref').split('/')[-1]
                title_id = ref_container[idx - 1].get('$ref').split('/')[-1]
                table_title = text_chunks_dict.get(title_id, {}).get("content")
                if table_title:
                    table_title = table_title.replace(" ", "_").lower()
                table_refs[table_id] = table_title or f"unknown_table_{table_id}"

        for node in table_container:
            table_data = node.get('data')

            if table_data:
                # Determine the type, falling back to 'text' if not specified
                node_label = node.get('label', 'text')
                
                # Use 'self_ref' for ID if available, otherwise use a counter
                node_id = node.get('self_ref', f"unknown_{len(tables)}").split('/')[-1]
                table_type = table_refs.get(node_id, f"unknown_table_{node_id}")

                rows = table_data.get('grid')
                if not rows or not isinstance(rows, list):
                    return []
                
                # Extract headers (first row)
                headers = [cell.get("text", "").strip() for cell in rows[0]]
                data_rows = rows[1:]

                table_rows = []
                for row in data_rows:
                    row_dict = {}
                    for idx, cell in enumerate(row):
                        key = headers[idx] if idx < len(headers) else f"col_{idx}"
                        value = cell.get("text", "").strip()
                        row_dict[key] = value

                    if table_type == "capital_calls":
                        parsed_row = self._parse_capital_call_row(row_dict)
                    elif table_type == "distributions":
                        parsed_row = self._parse_distribution_row(row_dict)
                    elif table_type == "adjustments":
                        parsed_row = self._parse_adjustment_row(row_dict)
                    else:
                        continue
                    
                    table_rows.append(parsed_row)
                
                tables[table_type] = {
                    "id": node_id,
                    "type": node_label,
                    "rows": table_rows
                }

        return tables   

    # --- Row Parsing Helper Methods (Slightly adjusted for robustness) ---

    def _parse_capital_call_row(self, row_map: Dict[str, Any]) -> Dict[str, Any]:
        """Parses and normalizes a single Capital Call row."""        
        # Required field validation
        if not row_map.get("Date") or not row_map.get("Amount"):
            raise ValueError("Missing required Date or Amount for Capital Call.")

        return {
            "date": self._clean_date(row_map.get("Date")),
            "call_type": row_map.get("Call Number", "").strip(),
            "amount": self._clean_amount(row_map.get("Amount")),
            "description": row_map.get("Description", "").strip().replace('\n', ' ')
        }

    def _parse_distribution_row(self, row_map: Dict[str, Any]) -> Dict[str, Any]:
        """Parses and normalizes a single Distribution row."""        
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

    def _parse_adjustment_row(self, row_map: Dict[str, Any]) -> Dict[str, Any]:
        """Parses and normalizes a single Adjustment row."""        
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
