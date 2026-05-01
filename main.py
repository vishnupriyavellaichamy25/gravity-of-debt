from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from explain import explain_prediction, load_pipeline

app = FastAPI(title="Gravity of Debt API", description="Credit Risk Prediction Engine", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline globally to avoid reloading on each request
pipeline_artifact = None

@app.on_event("startup")
def startup_event():
    global pipeline_artifact
    try:
        pipeline_artifact = load_pipeline()
        print("Model pipeline loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model. Error: {e}")

class ApplicantProfile(BaseModel):
    loan_amnt: float = Field(..., description="The listed amount of the loan applied for by the borrower.")
    int_rate: float = Field(..., description="Interest Rate on the loan")
    installment: float = Field(..., description="The monthly payment owed by the borrower")
    grade: str = Field(..., description="LC assigned loan grade")
    sub_grade: str = Field(..., description="LC assigned loan subgrade")
    emp_length: str = Field(..., description="Employment length in years. Possible values are between 0 and 10")
    home_ownership: str = Field(..., description="The home ownership status provided by the borrower during registration")
    annual_inc: float = Field(..., description="The self-reported annual income provided by the borrower during registration")
    verification_status: str = Field(..., description="Indicates if income was verified by LC")
    purpose: str = Field(..., description="A category provided by the borrower for the loan request")
    dti: float = Field(..., description="A ratio calculated using the borrower's total monthly debt payments on the total debt obligations")
    delinq_2yrs: float = Field(..., description="The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years")
    fico_range_low: float = Field(..., description="The lower boundary range the borrower's FICO at loan origination belongs to")
    fico_range_high: float = Field(..., description="The upper boundary range the borrower's FICO at loan origination belongs to")
    open_acc: float = Field(..., description="The number of open credit lines in the borrower's credit file")
    pub_rec: float = Field(..., description="Number of derogatory public records")
    revol_bal: float = Field(..., description="Total credit revolving balance")
    revol_util: float = Field(..., description="Revolving line utilization rate")
    total_acc: float = Field(..., description="The total number of credit lines currently in the borrower's credit file")

@app.get("/health")
def health_check():
    if pipeline_artifact is None:
        return {"status": "degraded", "message": "Model not loaded"}
    return {"status": "ok", "message": "API and Model are running"}

@app.post("/predict")
def predict_risk(profile: ApplicantProfile):
    if pipeline_artifact is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not available")
        
    try:
        # Convert Pydantic model to dictionary
        applicant_data = profile.dict()
        
        # Get explanation and prediction
        results = explain_prediction(applicant_data, pipeline_artifact)
        
        # We don't need to return the internal processed features or base value unless we want to
        # but the prompt requested: default_probability, risk_level, top_reasons, shap_values
        return {
            "default_probability": round(results["default_probability"], 4),
            "risk_level": results["risk_level"],
            "top_reasons": results["top_reasons"],
            "shap_values": results["shap_values"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
