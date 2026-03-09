"""
Document ID Generator

Generates clean, unique, stable document IDs from PDF files.
Supports multiple strategies for ID generation.
"""

import hashlib
import re
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime


class DocIdGenerator:
    """Generate clean document IDs from PDF files"""
    
    @staticmethod
    def from_content_hash(pdf_path: str, prefix: str = "doc") -> str:
        """
        Generate ID from file content hash (most stable)
        
        Example: doc_a3f5b2c8
        
        Pros:
        - Stable: Same file always gets same ID
        - Unique: Different files get different IDs
        - Deduplication: Detects duplicate files
        
        Cons:
        - Not human-readable
        - Changes if file content changes
        """
        with open(pdf_path, 'rb') as f:
            # Read first 1MB for speed (enough to be unique)
            content = f.read(1024 * 1024)
            hash_digest = hashlib.sha256(content).hexdigest()[:8]
        
        return f"{prefix}_{hash_digest}"
    
    @staticmethod
    def from_filename(pdf_path: str, max_length: int = 50) -> str:
        """
        Generate ID from sanitized filename (human-readable)
        
        Example: cbe_annual_report_2023_24
        
        Pros:
        - Human-readable
        - Easy to identify document
        
        Cons:
        - Not unique (same filename in different folders)
        - Changes if file renamed
        """
        filename = Path(pdf_path).stem
        
        # Sanitize: lowercase, replace spaces/special chars with underscore
        sanitized = re.sub(r'[^a-z0-9]+', '_', filename.lower())
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip('_')
        
        return sanitized
    
    @staticmethod
    def from_filename_with_hash(pdf_path: str, max_length: int = 40) -> str:
        """
        Generate ID from filename + short hash (best of both worlds)
        
        Example: cbe_annual_report_2023_a3f5b2c8
        
        Pros:
        - Human-readable prefix
        - Unique suffix
        - Stable (hash-based)
        
        Cons:
        - Slightly longer
        """
        filename = Path(pdf_path).stem
        
        # Sanitize filename
        sanitized = re.sub(r'[^a-z0-9]+', '_', filename.lower())
        sanitized = sanitized.strip('_')
        
        # Truncate filename part
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip('_')
        
        # Add short hash
        with open(pdf_path, 'rb') as f:
            content = f.read(1024 * 1024)
            hash_digest = hashlib.sha256(content).hexdigest()[:8]
        
        return f"{sanitized}_{hash_digest}"
    
    @staticmethod
    def from_uuid(prefix: str = "doc") -> str:
        """
        Generate random UUID-based ID (always unique)
        
        Example: doc_a3f5b2c8_4d7e_9f1a
        
        Pros:
        - Always unique
        - No collisions
        
        Cons:
        - Not human-readable
        - Not stable (different ID each time)
        - Can't deduplicate
        """
        short_uuid = str(uuid.uuid4())[:23].replace('-', '_')
        return f"{prefix}_{short_uuid}"
    
    @staticmethod
    def from_timestamp(pdf_path: str, prefix: str = "doc") -> str:
        """
        Generate ID from filename + timestamp
        
        Example: cbe_annual_report_20260309_040530
        
        Pros:
        - Human-readable
        - Sortable by time
        
        Cons:
        - Not stable (changes each time)
        - Can have collisions
        """
        filename = Path(pdf_path).stem
        sanitized = re.sub(r'[^a-z0-9]+', '_', filename.lower())
        sanitized = sanitized.strip('_')[:30]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{sanitized}_{timestamp}"
    
    @staticmethod
    def generate(
        pdf_path: str, 
        strategy: str = "filename_with_hash",
        prefix: str = "doc"
    ) -> str:
        """
        Generate document ID using specified strategy
        
        Args:
            pdf_path: Path to PDF file
            strategy: One of:
                - "content_hash": Hash-based (stable, unique)
                - "filename": Sanitized filename (readable)
                - "filename_with_hash": Filename + hash (recommended)
                - "uuid": Random UUID (always unique)
                - "timestamp": Filename + timestamp (sortable)
            prefix: Prefix for ID (default: "doc")
        
        Returns:
            Generated document ID
        """
        strategies = {
            "content_hash": DocIdGenerator.from_content_hash,
            "filename": DocIdGenerator.from_filename,
            "filename_with_hash": DocIdGenerator.from_filename_with_hash,
            "uuid": lambda path: DocIdGenerator.from_uuid(prefix),
            "timestamp": DocIdGenerator.from_timestamp,
        }
        
        if strategy not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Choose from: {list(strategies.keys())}"
            )
        
        generator = strategies[strategy]
        
        # Handle strategies that don't take pdf_path
        if strategy == "uuid":
            return generator(pdf_path)
        else:
            return generator(pdf_path)


def generate_doc_id(
    pdf_path: str,
    strategy: str = "filename_with_hash"
) -> str:
    """
    Convenience function to generate document ID
    
    Recommended strategy: "filename_with_hash"
    - Human-readable prefix from filename
    - Unique hash suffix
    - Stable (same file = same ID)
    
    Example:
        >>> generate_doc_id("data/CBE ANNUAL REPORT 2023-24.pdf")
        'cbe_annual_report_2023_24_a3f5b2c8'
    """
    return DocIdGenerator.generate(pdf_path, strategy=strategy)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python doc_id_generator.py <pdf_path> [strategy]")
        print("\nStrategies:")
        print("  - content_hash (stable, unique)")
        print("  - filename (readable)")
        print("  - filename_with_hash (recommended)")
        print("  - uuid (always unique)")
        print("  - timestamp (sortable)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else "filename_with_hash"
    
    doc_id = generate_doc_id(pdf_path, strategy=strategy)
    print(f"Generated ID: {doc_id}")
