from src.classifier import ClauseClassifier
from src.chunker import DocumentChunker
from src.agent import DocumentInspectorAgent

class ClauseLens:
    def __init__(self, model_name:str, policy:str):
        self.model_name = model_name
        self.policy = policy

    def run(self, document_path:str):
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(document_path)

        classifier = ClauseClassifier('../models/clause_bert.onnx')
        predictions = classifier.run(chunks)

        agent = DocumentInspectorAgent(self.model_name, self.policy)
        output = agent.run(chunks, predictions)
        return output