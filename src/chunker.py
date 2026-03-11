import nltk
from pypdf import PdfReader
import re

class DocumentChunker:
    def __init__(self):
        pass

    def chunk_document(self, document_path: str, chunk_size: int = 250):
        nltk.download('punkt_tab')

        if document_path.endswith("pdf"):
            reader = PdfReader(document_path)
            doc = "\n".join([page.extract_text() for page in reader.pages])
            doc = re.sub(r'\xa0', ' ', doc)

        elif document_path.endswith("docx"):
            with open(document_path, "r") as f:
                doc = f.read()
        else:
            raise Exception("Unsupported file format")

        sentence_list = nltk.sent_tokenize(doc) 
        chunks = []
        chunk = ""
        for sentence in sentence_list:
            if len(chunk.split()) + len(sentence.split()) > chunk_size:
                chunks.append(chunk)
                chunk = ""
            chunk = chunk + sentence
        return chunks