# app.py
import json
import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any

app = FastAPI(title="OncoPilot Multi-Cancer Similarity Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientPayload(BaseModel):
    cancer_type: str = Field(default="LUAD", description="Must be 'LUAD' or 'BRCA'")
    pathology: Dict[str, Any]
    genomics: Dict[str, Any]
    imaging: Dict[str, Any]
    clinical: Dict[str, Any]

class MatchRequest(BaseModel):
    patient: PatientPayload
    weights: Dict[str, float]

class AIRequest(BaseModel):
    prompt: str

def load_local_database(cancer_type: str) -> List[Dict]:
    # Look for the specific multi-cancer DB file inside the data folder
    db_path = f"data/tcga_cases_{cancer_type}.json"
    
    # Fallback for hackathon safety if they are still in the root folder
    if not os.path.exists(db_path):
        db_path = f"tcga_cases_{cancer_type}.json"
        
    # Absolute fallback to your original DB name if nothing else exists
    if not os.path.exists(db_path):
        db_path = "tcga_cases.json" 
        
    if not os.path.exists(db_path):
        raise HTTPException(status_code=500, detail=f"Database file for {cancer_type} missing. Run build_dataset.py.")
        
    with open(db_path, "r") as f:
        return json.load(f)

# ── Normalization helpers ─────────────────────────────────────────────────────
def normalize_smoking(val: Any) -> str:
    v = str(val).lower().strip()
    if any(x in v for x in ['current', 'heavy', 'active', 'lifelong']):
        return 'smoker'
    if any(x in v for x in ['former', 'ex-', 'ex ']):
        return 'former'
    if 'never' in v:
        return 'never'
    return v

def normalize_margin(val: Any) -> str:
    v = str(val).lower().strip()
    if 'positive' in v:
        return 'positive'
    if 'negative' in v:
        return 'negative'
    return v

# ── Scoring functions ─────────────────────────────────────────────────────────
def compute_categorical_score(val1: Any, val2: Any) -> float:
    return 1.0 if str(val1).strip().lower() == str(val2).strip().lower() else 0.0

def compute_numerical_score(val1: float, val2: float, max_variance: float) -> float:
    try:
        if val1 is None or val2 is None:
            return 0.5
        diff = abs(float(val1) - float(val2))
        return max(0.0, min(1.0, 1.0 - (diff / max_variance)))
    except (ValueError, TypeError):
        return 0.0

def compute_array_score(arr1: List, arr2: List) -> float:
    set1 = set([str(x).strip().lower() for x in arr1])
    set2 = set([str(x).strip().lower() for x in arr2])
    if not set1 or not set2:
        return 1.0 if set1 == set2 else 0.0
    return float(len(set1 & set2)) / float(max(len(set1), len(set2)))

@app.post("/api/match")
async def match_patient(payload: MatchRequest):
    # Dynamically load the database based on the selected cancer type!
    cancer_type = payload.patient.cancer_type
    database = load_local_database(cancer_type)
    
    p = payload.patient
    w = payload.weights

    total_w = sum([w.get(k, 25) for k in ["pathology","genomics","imaging","clinical"]])
    if total_w == 0: total_w = 1.0
    nw = {k: w.get(k, 25) / total_w for k in ["pathology","genomics","imaging","clinical"]}

    results = []
    for case in database:
        params = []

        # 1. Pathology
        p_sub = compute_categorical_score(p.pathology.get("subtype"), case["pathology"].get("subtype"))
        p_grd = compute_categorical_score(p.pathology.get("tumor_grade"), case["pathology"].get("tumor_grade"))
        p_mit = compute_categorical_score(p.pathology.get("mitotic_index"), case["pathology"].get("mitotic_index"))
        p_mrg = compute_categorical_score(
            normalize_margin(p.pathology.get("surgical_margin", "")),
            normalize_margin(case["pathology"].get("surgical_margin", ""))
        )
        p_siz = compute_numerical_score(p.pathology.get("tumor_size_mm", 0), case["pathology"].get("tumor_size_mm", 0), 100.0)
        path_avg = (p_sub + p_grd + p_mit + p_mrg + p_siz) / 5.0

        params.extend([
            {"name": "Cancer Subtype",   "patient": str(p.pathology.get("subtype", "N/A")),         "match": str(case["pathology"].get("subtype", "N/A")),        "score": "green" if p_sub > 0.8 else "red"},
            {"name": "Tumor Grade",      "patient": str(p.pathology.get("tumor_grade", "N/A")),      "match": str(case["pathology"].get("tumor_grade", "N/A")),    "score": "green" if p_grd > 0.8 else "red"},
            {"name": "Mitotic Index",    "patient": str(p.pathology.get("mitotic_index", "N/A")),    "match": str(case["pathology"].get("mitotic_index", "N/A")),  "score": "green" if p_mit > 0.8 else "red"},
            {"name": "Surgical Margins", "patient": str(p.pathology.get("surgical_margin", "N/A")),  "match": str(case["pathology"].get("surgical_margin", "N/A")),"score": "green" if p_mrg > 0.8 else "red"},
            {"name": "Tumor Size (mm)",  "patient": f"{p.pathology.get('tumor_size_mm', 0)}mm", "match": f"{case['pathology'].get('tumor_size_mm', 0)}mm","score": "green" if p_siz > 0.8 else "amber" if p_siz > 0.5 else "red"},
        ])

        # 2. Genomics
        g_drv = compute_categorical_score(p.genomics.get("driver_mutation"), case["genomics"].get("driver_mutation"))
        g_sec = compute_categorical_score(p.genomics.get("secondary_mutation"), case["genomics"].get("secondary_mutation"))
        g_tmb = compute_numerical_score(p.genomics.get("tmb", 0), case["genomics"].get("tmb", 0), 30.0)
        g_pdl = compute_numerical_score(p.genomics.get("pdl1_percent", 0), case["genomics"].get("pdl1_percent", 0), 100.0)
        g_cnv = compute_categorical_score(p.genomics.get("cnv"), case["genomics"].get("cnv"))
        gen_avg = (g_drv + g_sec + g_tmb + g_pdl + g_cnv) / 5.0

        params.extend([
            {"name": "Driver Mutation",         "patient": str(p.genomics.get("driver_mutation", "N/A")),   "match": str(case["genomics"].get("driver_mutation", "N/A")),  "score": "green" if g_drv > 0.8 else "red"},
            {"name": "Secondary Mutation",      "patient": str(p.genomics.get("secondary_mutation", "N/A")),"match": str(case["genomics"].get("secondary_mutation", "N/A")),"score": "green" if g_sec > 0.8 else "red"},
            {"name": "Tumor Mutational Burden", "patient": f"{p.genomics.get('tmb', 0)} mut/Mb",       "match": f"{case['genomics'].get('tmb', 0)} mut/Mb",        "score": "green" if g_tmb > 0.8 else "amber" if g_tmb > 0.5 else "red"},
            {"name": "PD-L1 / ER Expression",   "patient": f"{p.genomics.get('pdl1_percent', 0)}%",    "match": f"{case['genomics'].get('pdl1_percent', 0)}%",     "score": "green" if g_pdl > 0.8 else "amber" if g_pdl > 0.5 else "red"},
            {"name": "Copy Number Variation",   "patient": str(p.genomics.get("cnv", "N/A")),              "match": str(case["genomics"].get("cnv", "N/A")),               "score": "green" if g_cnv > 0.8 else "red"},
        ])

        # 3. Imaging
        i_lob = compute_categorical_score(p.imaging.get("lobe"), case["imaging"].get("lobe"))
        i_den = compute_categorical_score(p.imaging.get("density"), case["imaging"].get("density"))
        i_nst = compute_categorical_score(p.imaging.get("n_stage"), case["imaging"].get("n_stage"))
        i_ple = compute_categorical_score(p.imaging.get("pleural_invasion"), case["imaging"].get("pleural_invasion"))
        i_met = compute_array_score(p.imaging.get("metastasis_sites", []), case["imaging"].get("metastasis_sites", []))
        img_avg = (i_lob + i_den + i_nst + i_ple + i_met) / 5.0

        params.extend([
            {"name": "Anatomical Location",   "patient": str(p.imaging.get("lobe", "N/A")),              "match": str(case["imaging"].get("lobe", "N/A")),             "score": "green" if i_lob > 0.8 else "red"},
            {"name": "Radiographic Density",  "patient": str(p.imaging.get("density", "N/A")),           "match": str(case["imaging"].get("density", "N/A")),          "score": "green" if i_den > 0.8 else "red"},
            {"name": "Nodal Involvement (N)", "patient": str(p.imaging.get("n_stage", "N/A")),           "match": str(case["imaging"].get("n_stage", "N/A")),          "score": "green" if i_nst > 0.8 else "red"},
            {"name": "Pleural Invasion",      "patient": str(p.imaging.get("pleural_invasion", "N/A")),  "match": str(case["imaging"].get("pleural_invasion", "N/A")), "score": "green" if i_ple > 0.8 else "red"},
            {"name": "Metastasis Sites",      "patient": ", ".join(p.imaging.get("metastasis_sites",[])), "match": ", ".join(case["imaging"].get("metastasis_sites", [])), "score": "green" if i_met > 0.8 else "amber" if i_met > 0.2 else "red"},
        ])

        # 4. Clinical
        c_age = compute_numerical_score(p.clinical.get("age", 0), case["clinical"].get("age", 0), 40.0)
        c_sex = compute_categorical_score(p.clinical.get("sex"), case["clinical"].get("sex"))
        c_smk = compute_categorical_score(
            normalize_smoking(p.clinical.get("smoking_history", "")),
            normalize_smoking(case["clinical"].get("smoking_history", ""))
        )
        c_ecg = compute_numerical_score(p.clinical.get("ecog_status", 0), case["clinical"].get("ecog_status", 0), 4.0)
        c_cmb = compute_array_score(p.clinical.get("co_morbidities", []), case["clinical"].get("co_morbidities", []))
        cli_avg = (c_age + c_sex + c_smk + c_ecg + c_cmb) / 5.0

        params.extend([
            {"name": "Patient Age",              "patient": f"{p.clinical.get('age', 0)} yrs",              "match": f"{case['clinical'].get('age', 0)} yrs",         "score": "green" if c_age > 0.8 else "amber" if c_age > 0.5 else "red"},
            {"name": "Biological Sex",           "patient": str(p.clinical.get("sex", "N/A")),                  "match": str(case["clinical"].get("sex", "N/A")),             "score": "green" if c_sex > 0.8 else "red"},
            {"name": "Tobacco Exposure History", "patient": str(p.clinical.get("smoking_history", "N/A")),      "match": str(case["clinical"].get("smoking_history", "N/A")), "score": "green" if c_smk > 0.8 else "red"},
            {"name": "ECOG Performance Scale",   "patient": f"ECOG {p.clinical.get('ecog_status', 0)}",     "match": f"ECOG {case['clinical'].get('ecog_status', 0)}", "score": "green" if c_ecg > 0.8 else "amber" if c_ecg > 0.4 else "red"},
            {"name": "Documented Comorbidities", "patient": ", ".join(p.clinical.get("co_morbidities",[])), "match": ", ".join(case["clinical"].get("co_morbidities", [])), "score": "green" if c_cmb > 0.8 else "amber" if c_cmb > 0.2 else "red"},
        ])

        score = (path_avg*nw["pathology"] + gen_avg*nw["genomics"] + img_avg*nw["imaging"] + cli_avg*nw["clinical"]) * 100.0
        results.append({
            "patient_id": case["patient_id"],
            "cancer_type": case.get("cancer_type", "LUAD"), # Safety fallback
            "similarity_score": round(score, 1),
            "treatment_history": case["treatment_history"],
            "guideline_citation": case["guideline_citation"],
            "outcome": case["outcome"],
            "stage": case["clinical"].get("stage", "Unknown"),
            "parameters": params,
            "raw_case_data": case,
        })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:5]

@app.post("/api/ai-rationale")
async def ai_rationale(req: AIRequest):
    """Proxy to avoid CORS — set ANTHROPIC_API_KEY env var before running."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set.")
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-localhost:80014-20250514", "max_tokens": 1000,
                  "messages": [{"role": "user", "content": req.prompt}]}
        )
    return r.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)