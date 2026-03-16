from src.classifier import ClauseClassifier
from src.chunker import DocumentChunker
from src.agent import DocumentInspectorAgent
import json

class ClauseLens:
    def __init__(self, classifier_path, model_name:str, policy:str, chunk_size: int):
        self.model_name = model_name
        self.policy = policy
        self.chunk_size = chunk_size
        self.classifier_path = classifier_path
    
    def run(self, document_file, document_type):
        
        #clause lens pipeline: chunking -> clause classification -> agent -> json

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(document_file, self.chunk_size, document_type)

        classifier = ClauseClassifier(self.classifier_path)
        predictions = classifier.run(chunks)
        agent = DocumentInspectorAgent(self.model_name, self.policy)
        output = agent.run(chunks, predictions)

        json_output = {"output": [json.loads(output[i]) for i in range(len(output))]}
        return json_output