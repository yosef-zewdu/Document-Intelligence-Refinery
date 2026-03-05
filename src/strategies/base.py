from abc import ABC, abstractmethod
from src.models import DocumentProfile, ExtractedDocument

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        """
        Extracts content from a PDF based on the document profile.
        Returns an ExtractedDocument.
        """
        pass

    @abstractmethod
    def get_confidence_score(self, extracted: ExtractedDocument) -> float:
        """
        Retrieves or calculates a confidence score (0.0 to 1.0) for the extraction.
        """
        pass
