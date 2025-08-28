"""Custom Semantic Scholar Tool to get complete citations."""

import re
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun

class CustomSemanticScholarAPIWrapper(SemanticScholarAPIWrapper):
    """
    Wrapper around semanticscholar.org API that returns complete citations.
    """

    def __init__(self):
        super().__init__()
        self.returned_fields = [
            "title",
            "abstract",
            "venue",
            "year",
            "paperId",
            "citationCount",
            "openAccessPdf",
            "authors",
            "externalIds",
            "journal",
        ]

    def run(self, query: str) -> str:
        """Run the Semantic Scholar API and get complete citations."""
        
        # Check if the query is for a specific author
        author_match = re.match(r"(?:papers by|author:|from)\s+(.*)", query, re.IGNORECASE)
        if author_match:
            author_name = author_match.group(1).strip()
            try:
                from semanticscholar import SemanticScholar
                sch = SemanticScholar()
                authors = sch.search_author(author_name)
                if not authors or not authors[0]:
                    return f"Could not find author: {author_name}"
                
                # Assume the first result is the correct author
                author_id = authors[0].authorId
                results = sch.get_author_papers(author_id, limit=self.load_max_docs, fields=self.returned_fields)
            except Exception as e:
                return f"An error occurred during author search: {e}"
        else:
            results = self.semanticscholar_search(
                query, limit=self.load_max_docs, fields=self.returned_fields
            )

        documents = []
        for item in results[: self.top_k_results]:
            authors = ", ".join(
                author["name"] for author in getattr(item, "authors", [])
            )
            journal_name = getattr(getattr(item, "journal", {}), "name", None)
            journal_volume = getattr(getattr(item, "journal", {}), "volume", None)
            journal_pages = getattr(getattr(item, "journal", {}), "pages", None)
            doi = getattr(item, "externalIds", {}).get("DOI", None)
            
            citation = f"Authors: {authors}\n"
            citation += f"Year: {getattr(item, 'year', None)}\n"
            citation += f"Title: {getattr(item, 'title', None)}\n"
            if journal_name:
                citation += f"Journal: {journal_name}\n"
            if journal_volume:
                citation += f"Volume: {journal_volume}\n"
            if journal_pages:
                citation += f"Pages: {journal_pages}\n"
            if doi:
                citation += f"DOI: {doi}\n"
            citation += f"Abstract: {getattr(item, 'abstract', None)}\n"
            documents.append(citation)

        if documents:
            return "\n\n".join(documents)[: self.doc_content_chars_max]
        else:
            return "No results found."

class CustomSemanticScholarQueryRun(SemanticScholarQueryRun):
    """Tool that searches the semanticscholar API with a custom wrapper."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Lazily initialize the wrapper to avoid issues on startup
        self.api_wrapper = None

    def _run(self, query: str) -> str:
        """Run the tool."""
        if self.api_wrapper is None:
            self.api_wrapper = CustomSemanticScholarAPIWrapper()
        return self.api_wrapper.run(query)


