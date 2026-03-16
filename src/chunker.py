import nltk
from pypdf import PdfReader
import re
from io import TextIOWrapper

class DocumentChunker:
    def __init__(self):
        pass

    def chunk_document(self, document_file, chunk_size: int, doc_type: str):
        
        nltk.download('punkt_tab')

        if doc_type == "pdf":
            reader = PdfReader(document_file)
            doc = "\n".join([page.extract_text() for page in reader.pages])
            doc = re.sub(r'\xa0', ' ', doc)

        elif doc_type == "txt":
            doc = document_file.read().decode("utf-8", errors="replace")
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