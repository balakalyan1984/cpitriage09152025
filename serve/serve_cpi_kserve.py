import os, tarfile, tempfile, joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# KServe storage initializer will download to /mnt/models
MODEL_FILE = "model.pkl"
MOUNT_DIR  = "/mnt/models"
MODEL_PATH = os.path.join(MOUNT_DIR, MODEL_FILE)
TGZ_PATH   = MODEL_PATH + ".tgz"

def _load_pipe():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    if os.path.exists(TGZ_PATH):
        with tarfile.open(TGZ_PATH, "r:*") as tar, tempfile.TemporaryDirectory() as tmp:
            m = next((m for m in tar.getmembers() if m.name.endswith(".pkl")), None)
            if not m:
                raise RuntimeError("No .pkl inside model.pkl.tgz")
            tar.extract(m, tmp)
            return joblib.load(os.path.join(tmp, m.name))
    raise FileNotFoundError(f"Model not found at {MODEL_PATH} or {TGZ_PATH}")

pipe = _load_pipe()
app = FastAPI(title="CPI Logs Classifier", version="1.0")

class Instance(BaseModel):
    ARTIFACT_NAME: Optional[str] = ""
    ORIGIN_COMPONENT_NAME: Optional[str] = ""
    LOG_LEVEL: Optional[str] = ""

class PredictRequest(BaseModel):
    instances: List[Instance]

@app.get("/v2/greet")
def greet():
    return {"status": "ok", "model": "loaded", "checked": [MODEL_PATH, TGZ_PATH]}

@app.post("/v2/predict")
def predict(req: PredictRequest):
    records = [{
        "ARTIFACT_NAME": (x.ARTIFACT_NAME or ""),
        "ORIGIN_COMPONENT_NAME": (x.ORIGIN_COMPONENT_NAME or ""),
        "LOG_LEVEL": (x.LOG_LEVEL or ""),
    } for x in req.instances]
    preds = pipe.predict(records)
    return {"predictions": [str(p) for p in preds]}
