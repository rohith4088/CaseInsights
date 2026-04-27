import io, os, json, sys

# Make imports work both locally (python3 main.py) and in Docker (uvicorn backend.main:app)
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
from predictor import CategoryPredictor
from bert_trainer import BERTTrainer

app = FastAPI(title="AI Issue Categorization Dashboard API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")

@app.get("/")
def root():
    return RedirectResponse(url="/app/")

print("Initializing AI Predictor...")
try:
    predictor = CategoryPredictor()
    print("Predictor ready.")
except Exception as e:
    print(f"Predictor init error: {e}")
    predictor = None

BERT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "bert_model")
print("Initializing BERT engine...")
try:
    bert_trainer = BERTTrainer(BERT_MODEL_DIR)
    if bert_trainer.is_trained:
        print("Found saved BERT model — loading...")
        bert_trainer.load()
except Exception as e:
    print(f"BERT init error: {e}")
    bert_trainer = None

# In-memory: case details + combined_text (for active learning)
current_dataset: dict = {}
current_df_ref = None   # reference to last uploaded DataFrame for active learning retraining
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "feedback.json")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global current_dataset, current_df_ref
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.csv')):
        raise HTTPException(status_code=400, detail="Upload an .xlsx or .csv file.")
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') \
            else pd.read_excel(io.BytesIO(contents))
        result = predictor.predict(df)
        # Store dataset: include Combined_Text for active learning
        df_full = df.fillna("")
        current_df_ref = df_full.copy()
        current_dataset = {}
        for i, (_, row) in enumerate(df_full.iterrows()):
            key = str(row.get('Case Number', i))
            current_dataset[key] = {k: str(v) for k, v in row.items()}
            # Also include the combined text that predictor generated
            if 'Combined_Text' in df_full.columns:
                current_dataset[key]['_combined_text'] = str(row.get('Combined_Text', ''))
        return result
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
def get_categories():
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized.")
    return {"categories": predictor.valid_categories}


@app.get("/api/case/{case_number}")
def get_case(case_number: str):
    row = current_dataset.get(case_number)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found in current session.")
    return row


class FeedbackPayload(BaseModel):
    case_number: str
    corrected_category: str


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackPayload):
    """Save correction and trigger active learning retrain."""
    try:
        # Look up the case's combined text for active learning
        case = current_dataset.get(payload.case_number, {})
        combined_text = case.get('_combined_text', '')

        existing = []
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH, 'r') as f:
                existing = json.load(f)

        existing.append({
            "case_number": payload.case_number,
            "corrected_category": payload.corrected_category,
            "text": combined_text
        })
        with open(FEEDBACK_PATH, 'w') as f:
            json.dump(existing, f, indent=2)

        # Trigger active learning retrain in the background
        retrain_result = {"status": "saved_no_retrain"}
        if predictor and combined_text:
            retrain_result = predictor.retrain_with_feedback(FEEDBACK_PATH, current_df_ref)

        return {
            "status": "saved",
            "total_feedback": len(existing),
            "retrain": retrain_result
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/retrain")
def force_retrain():
    """Force a full retrain incorporating all saved feedback."""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized.")
    result = predictor.retrain_with_feedback(FEEDBACK_PATH, current_df_ref)
    return result


@app.get("/api/feedback/stats")
def feedback_stats():
    """Return stats about accumulated human feedback."""
    if not os.path.exists(FEEDBACK_PATH):
        return {"total": 0, "categories": {}}
    with open(FEEDBACK_PATH) as f:
        data = json.load(f)
    from collections import Counter
    cats = Counter(fb['corrected_category'] for fb in data if fb.get('corrected_category'))
    return {"total": len(data), "categories": dict(cats.most_common(10))}


# -----------------------------------------------------------------------
# BERT endpoints
# -----------------------------------------------------------------------

@app.post("/api/bert/train")
def bert_train():
    """Start BERT fine-tuning in the background. Uses current_dataset texts."""
    if bert_trainer is None:
        raise HTTPException(status_code=500, detail="BERT engine not available.")
    if not current_dataset:
        raise HTTPException(status_code=400, detail="Upload a dataset first.")
    if bert_trainer.status == "training":
        return {"status": "already_training"}

    # Build texts + labels from current dataset
    texts, labels = [], []
    for case in current_dataset.values():
        text = case.get("_combined_text", "")
        # Use the AI-predicted category as the label (supervised from LinearSVC results)
        label = case.get("Predicted_Category", "") or case.get("Other Issue Category Description", "")
        if text and label and label not in ("", "Others", "nan"):
            texts.append(text)
            labels.append(label)

    if len(texts) < 50:
        raise HTTPException(status_code=400, detail=f"Only {len(texts)} usable labeled rows. Need at least 50.")

    result = bert_trainer.start_training_async(texts, labels, epochs=3, batch_size=16)
    return result


@app.get("/api/bert/status")
def bert_status():
    """Poll BERT training progress."""
    if bert_trainer is None:
        return {"status": "unavailable", "progress": 0, "is_trained": False}
    return bert_trainer.get_status()


@app.post("/api/bert/predict")
def bert_predict():
    """Re-run predictions on current dataset using the fine-tuned BERT model."""
    if bert_trainer is None or not bert_trainer.is_trained:
        raise HTTPException(status_code=400, detail="BERT model not trained yet.")
    if not current_dataset:
        raise HTTPException(status_code=400, detail="Upload a dataset first.")
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized.")

    try:
        cases = list(current_dataset.values())
        texts = [c.get("_combined_text", c.get("Subject", "")) for c in cases]
        labels, confs, top3s = bert_trainer.predict(texts)

        # Patch current_dataset with BERT predictions
        for i, case in enumerate(cases):
            case["Predicted_Category"] = labels[i]
            case["Prediction_Confidence"] = str(round(confs[i], 3))

        # Re-run analytics using predictor's existing methods on patched data
        import pandas as pd
        df = pd.DataFrame(cases)
        df["Predicted_Category"] = labels
        df["Prediction_Confidence"] = confs
        if predictor.cat_to_main:
            df["Main_Category"] = df["Predicted_Category"].map(
                lambda x: predictor.cat_to_main.get(x, "Others"))
        df = df.fillna("")

        # Summary
        n = len(df)
        top_cats = df["Predicted_Category"].value_counts().head(10)
        main_counts = df["Main_Category"].value_counts() if "Main_Category" in df.columns else pd.Series()
        summary = {
            "top_categories": {"labels": top_cats.index.tolist(), "values": top_cats.values.tolist()},
            "main_categories": {"labels": main_counts.index.tolist(), "values": main_counts.values.tolist()},
            "total_cases": n,
            "high_confidence": sum(1 for c in confs if c >= 0.8),
            "low_confidence": sum(1 for c in confs if c < 0.5),
            "engine": "DistilBERT"
        }

        # Table
        cols = ["Case Number", "Subject", "Severity", "Priority",
                "Predicted_Category", "Prediction_Confidence", "Main_Category",
                "Case Owner", "Opened Date", "Status"]
        table_data = []
        for i, case in enumerate(cases):
            row = {c: case.get(c, "") for c in cols if c in case}
            row["top_3"] = top3s[i]
            table_data.append(row)

        return {"table": table_data, "summary": summary, "engine": "bert",
                "valid_categories": predictor.valid_categories}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
