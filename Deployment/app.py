from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI(title="AI Text Detection API", version="1.0")

MODEL_PATH = "Hamza-003/ai-text-detector-roberta"
model = None
tokenizer = None
device = "cpu"

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    print("Loading model from Hugging Face Hub...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()

@app.post("/predict")
def predict(request: TextRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty text provided")

    inputs = tokenizer(
        request.text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    labels = ["Human", "AI"]
    result = labels[pred_idx]

    return {
        "prediction": result,
        "confidence_score": round(confidence, 4),
        "probabilities": {
            "human": round(probs[0][0].item(), 4),
            "ai": round(probs[0][1].item(), 4)
        }
    }

# Health Check Endpoint
@app.get("/")
def home():
    return {"status": "running", "model": "RoBERTa-FineTuned"}