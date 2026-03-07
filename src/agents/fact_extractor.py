"""
Enhanced Fact Table Extractor
Phase 4: Extract key-value facts for precise querying

Features:
- Extract financial/numerical facts from chunks
- Store with full provenance (doc_id, page_num, bbox, content_hash)
- Support complex queries with SQL
- Handle different data types (financial, dates, percentages, etc.)
"""

import sqlite3
import os
import re
import json
from typing import List, Dict, Any, Optional
from src.models.types import LDU, TableStructure, BBox, ProvenanceChain


class EnhancedFactTableExtractor:
    """
    Enhanced fact extractor with better parsing and provenance tracking
    """
    
    def __init__(self, db_path: str = ".refinery/db/fact_table.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop old table if exists (for migration)
        # cursor.execute("DROP TABLE IF EXISTS facts")
        
        # Create enhanced facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                entity TEXT NOT NULL,
                value TEXT NOT NULL,
                value_numeric REAL,
                data_type TEXT NOT NULL,
                unit TEXT,
                context TEXT,
                bbox_json TEXT,
                content_hash TEXT,
                parent_section TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity ON facts(entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_page ON facts(doc_id, page_num)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_type ON facts(data_type)")
        
        conn.commit()
        conn.close()
    
    def _extract_numeric_value(self, value_str: str) -> Optional[float]:
        """Extract numeric value from string"""
        # Remove common separators and extract number
        cleaned = re.sub(r'[,\s]', '', value_str)
        match = re.search(r'[-+]?\d*\.?\d+', cleaned)
        if match:
            try:
                return float(match.group())
            except:
                return None
        return None
    
    def _detect_data_type(self, entity: str, value: str) -> str:
        """Detect data type from entity and value"""
        entity_lower = entity.lower()
        value_lower = value.lower()
        
        # Financial indicators
        if any(term in entity_lower for term in ['revenue', 'income', 'expense', 'profit', 'loss', 'asset', 'liability', 'equity', 'capital', 'deposit', 'loan']):
            return 'financial'
        
        # Date indicators
        if any(term in entity_lower for term in ['date', 'year', 'period', 'quarter', 'month']):
            return 'date'
        
        # Percentage indicators
        if '%' in value or 'percent' in value_lower:
            return 'percentage'
        
        # Ratio indicators
        if any(term in entity_lower for term in ['ratio', 'rate', 'margin']):
            return 'ratio'
        
        # Count indicators
        if any(term in entity_lower for term in ['count', 'number', 'total', 'quantity']):
            return 'count'
        
        return 'general'
    
    def _extract_unit(self, value: str) -> Optional[str]:
        """Extract unit from value string"""
        # Common units
        units = ['ETB', 'USD', 'EUR', 'birr', 'million', 'billion', 'thousand', '%', 'percent']
        value_lower = value.lower()
        
        for unit in units:
            if unit.lower() in value_lower:
                return unit
        
        return None
    
    def ingest_from_chunks(self, doc_id: str, chunks: List[LDU]):
        """
        Extract facts from LDU chunks
        
        Args:
            doc_id: Document identifier
            chunks: List of LDU chunks
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        facts_extracted = 0
        
        for chunk in chunks:
            # Skip non-text chunks for now
            if chunk.chunk_type not in ['text', 'table', 'heading']:
                continue
            
            # Extract facts from content
            facts = self._extract_facts_from_content(
                content=chunk.content,
                doc_id=doc_id,
                page_refs=chunk.page_refs,
                bbox=chunk.bounding_box,
                content_hash=chunk.content_hash,
                parent_section=chunk.parent_section,
                chunk_type=chunk.chunk_type
            )
            
            # Insert facts
            for fact in facts:
                cursor.execute("""
                    INSERT INTO facts (
                        doc_id, page_num, entity, value, value_numeric,
                        data_type, unit, context, bbox_json, content_hash,
                        parent_section, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fact['doc_id'],
                    fact['page_num'],
                    fact['entity'],
                    fact['value'],
                    fact.get('value_numeric'),
                    fact['data_type'],
                    fact.get('unit'),
                    fact.get('context'),
                    fact.get('bbox_json'),
                    fact.get('content_hash'),
                    fact.get('parent_section'),
                    fact.get('metadata_json')
                ))
                facts_extracted += 1
        
        conn.commit()
        conn.close()
        
        return facts_extracted
    
    def _extract_facts_from_content(
        self,
        content: str,
        doc_id: str,
        page_refs: List[int],
        bbox: Optional[BBox],
        content_hash: Optional[str],
        parent_section: Optional[str],
        chunk_type: str
    ) -> List[Dict[str, Any]]:
        """Extract facts from chunk content"""
        facts = []
        
        # Pattern 1: "Entity: Value" or "Entity | Value"
        pattern1 = r'([^:\|\n]+)[\:\|]\s*([^\n]+)'
        matches = re.findall(pattern1, content)
        
        for entity, value in matches:
            entity = entity.strip()
            value = value.strip()
            
            # Skip if too short or too long
            if len(entity) < 3 or len(entity) > 200 or len(value) < 1 or len(value) > 500:
                continue
            
            # Skip if entity looks like a sentence
            if entity.count(' ') > 10:
                continue
            
            # Detect data type
            data_type = self._detect_data_type(entity, value)
            
            # Extract numeric value
            value_numeric = self._extract_numeric_value(value)
            
            # Extract unit
            unit = self._extract_unit(value)
            
            # Create fact
            fact = {
                'doc_id': doc_id,
                'page_num': page_refs[0] if page_refs else 1,
                'entity': entity,
                'value': value,
                'value_numeric': value_numeric,
                'data_type': data_type,
                'unit': unit,
                'context': content[:200],  # First 200 chars as context
                'bbox_json': json.dumps(bbox.model_dump()) if bbox else None,
                'content_hash': content_hash,
                'parent_section': parent_section,
                'metadata_json': json.dumps({'chunk_type': chunk_type})
            }
            
            facts.append(fact)
        
        # Pattern 2: Financial statements (numbers with labels)
        # Example: "Interest income    101,040,098,062"
        pattern2 = r'([A-Za-z][A-Za-z\s]+?)\s+([\d,\.]+(?:\s*(?:ETB|USD|birr|million|billion))?)'
        matches = re.findall(pattern2, content)
        
        for entity, value in matches:
            entity = entity.strip()
            value = value.strip()
            
            # Skip if already captured or invalid
            if len(entity) < 5 or len(entity) > 100:
                continue
            
            # Must have numeric value
            value_numeric = self._extract_numeric_value(value)
            if value_numeric is None:
                continue
            
            # Detect data type
            data_type = self._detect_data_type(entity, value)
            
            # Extract unit
            unit = self._extract_unit(value)
            
            # Create fact
            fact = {
                'doc_id': doc_id,
                'page_num': page_refs[0] if page_refs else 1,
                'entity': entity,
                'value': value,
                'value_numeric': value_numeric,
                'data_type': data_type,
                'unit': unit,
                'context': content[:200],
                'bbox_json': json.dumps(bbox.model_dump()) if bbox else None,
                'content_hash': content_hash,
                'parent_section': parent_section,
                'metadata_json': json.dumps({'chunk_type': chunk_type})
            }
            
            facts.append(fact)
        
        return facts
    
    def query_facts(
        self,
        entity_query: Optional[str] = None,
        doc_id: Optional[str] = None,
        data_type: Optional[str] = None,
        page_num: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query facts with filters
        
        Args:
            entity_query: Search term for entity (LIKE query)
            doc_id: Filter by document ID
            data_type: Filter by data type
            page_num: Filter by page number
            limit: Maximum results
        
        Returns:
            List of facts with provenance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM facts WHERE 1=1"
        params = []
        
        if entity_query:
            query += " AND entity LIKE ?"
            params.append(f"%{entity_query}%")
        
        if doc_id:
            query += " AND doc_id = ?"
            params.append(doc_id)
        
        if data_type:
            query += " AND data_type = ?"
            params.append(data_type)
        
        if page_num:
            query += " AND page_num = ?"
            params.append(page_num)
        
        query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to dicts with provenance
        results = []
        for row in rows:
            fact = dict(zip(columns, row))
            
            # Add provenance
            fact['provenance'] = ProvenanceChain(
                document_name=fact['doc_id'],
                page_number=fact['page_num'],
                bbox=BBox(**json.loads(fact['bbox_json'])) if fact.get('bbox_json') else None,
                content_hash=fact.get('content_hash', '')
            ).model_dump()
            
            results.append(fact)
        
        return results
    
    def execute_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        Execute custom SQL query (SELECT only)
        
        Args:
            sql_query: SQL SELECT query
        
        Returns:
            Query results
        """
        # Security: only allow SELECT
        if not sql_query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the fact table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total facts
        cursor.execute("SELECT COUNT(*) FROM facts")
        total = cursor.fetchone()[0]
        
        # By data type
        cursor.execute("SELECT data_type, COUNT(*) FROM facts GROUP BY data_type")
        by_type = dict(cursor.fetchall())
        
        # By document
        cursor.execute("SELECT doc_id, COUNT(*) FROM facts GROUP BY doc_id")
        by_doc = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_facts': total,
            'by_data_type': by_type,
            'by_document': by_doc
        }


def main():
    """Example usage"""
    
    # Initialize extractor
    extractor = EnhancedFactTableExtractor()
    
    # Load chunks
    import json
    with open(".refinery/chunks/CBE ANNUAL REPORT 2023-24.pdf_chunks.json", 'r') as f:
        chunks_data = json.load(f)
    
    from src.models.types import LDU
    chunks = [LDU.model_validate(c) for c in chunks_data]
    
    print("Extracting facts from chunks...")
    facts_count = extractor.ingest_from_chunks("CBE ANNUAL REPORT 2023-24.pdf", chunks)
    print(f"✓ Extracted {facts_count} facts")
    
    # Get stats
    stats = extractor.get_stats()
    print(f"\nFact Table Statistics:")
    print(f"  Total facts: {stats['total_facts']}")
    print(f"  By type: {stats['by_data_type']}")
    
    # Query examples
    print(f"\nExample queries:")
    
    # Query 1: Interest income
    results = extractor.query_facts(entity_query="interest income", limit=3)
    print(f"\n1. Interest income facts: {len(results)} found")
    for r in results[:2]:
        print(f"   - {r['entity']}: {r['value']} (Page {r['page_num']})")
    
    # Query 2: Financial data
    results = extractor.query_facts(data_type="financial", limit=5)
    print(f"\n2. Financial facts: {len(results)} found")
    for r in results[:3]:
        print(f"   - {r['entity']}: {r['value']} (Page {r['page_num']})")


if __name__ == "__main__":
    main()
