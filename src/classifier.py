import onnxruntime
import numpy as np
from transformers import BertTokenizer

class ClauseClassifier:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = onnxruntime.InferenceSession(model_path)
        self.classes = [
            'Cap On Liability',
            'Other', 
            'Audit Rights',
            'Governing Law',
            'Exclusivity',
            'Ip Ownership Assignment', 
            'Non-Compete',
            'Termination For Convenience', 
            'Renewal Term',
            'Liquidated Damages'
        ]

    def run(self, chunks):
        predictions = []
    
        for chunk in chunks:
            tokenized_chunk = self.tokenizer(
                chunk,
                add_special_tokens=True,
                truncation = True,
                padding = 'max_length',
                max_length = 512,
                return_tensors = 'np'
            )

            cls_output = self.model.run(None, {
                'input_ids': tokenized_chunk['input_ids'].astype(np.int64),
                'attention_mask': tokenized_chunk['attention_mask'].astype(np.int64)
            })

            predictions.append(self.classes[np.argmax(cls_output[0])])
       
        return predictions
