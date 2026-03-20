from fastapi import FastAPI, UploadFile, File, HTTPException
import json

from src.clens import ClauseLens

app = FastAPI(title="ClauseLens")

with open('config/config.json') as f:
    config = json.load(f)

clens = ClauseLens(
    classifier_path=config["classifier_path"], 
    model_name=config["llm"], 
    policy=config["policy"], 
    chunk_size=config["chunk_size"]
)

@app.post("/analyze/")
def analyze_document(file: UploadFile = File(...)):
    doctype = file.filename.split('.')[-1].lower()
    
    if doctype not in ["pdf", "txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file format: use pdf or txt files")

    try:
        output = clens.run(file.file, doctype)
        return {
            "filename": file.filename,
            "results": output
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()