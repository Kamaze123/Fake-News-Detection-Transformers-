from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()


tokenizer = AutoTokenizer.from_pretrained("./fake_news_model")
model = AutoModelForSequenceClassification.from_pretrained("./fake_news_model")
model.eval()

class Statement(BaseModel):
    text: str

@app.post("/predict")
def predict(statement: Statement):
    inputs = tokenizer(
        statement.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    
    return {
        "statement": statement.text,
        "prediction": "FAKE" if pred == 0 else "REAL",
        "confidence": round(confidence, 3)
    }