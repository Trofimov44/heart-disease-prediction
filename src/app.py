import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field, EmailStr
import shap

loaded_model = CatBoostClassifier()
loaded_model.load_model("models/catboost_model.cbm")

explainer = shap.TreeExplainer(loaded_model)

df = pd.read_csv("data/heart_cleveland_upload_FE.csv")
df = df.drop('Unnamed: 0', axis=1)  
df = df.drop('condition', axis=1)

FEATURE_COLUMNS = df.columns.tolist()
app = FastAPI()


class Patient(BaseModel):
    age: int = Field(ge=0, lt=130, default=int(df["age"].median()))
    sex: int = Field(ge=0, default=1, le=1)
    cp:  int = Field(ge=0, default=0, le=3)
    trestbps: int = Field(ge=0, default=int(df["trestbps"].median()), le=300)
    chol:int = Field(ge=0, default=int(df["chol"].median()), le=1000)
    fbs: int = Field(ge=0, default=0, le=1)
    restecg: int = Field(ge=0, default=0, le=2)
    thalach: int = Field(ge=0, default=int(df["thalach"].median()), le=250)
    exang: int = Field(ge=0, default=0, le=1)
    oldpeak: float = Field(ge=0, default= df["oldpeak"].median(), le=10)
    slope: int = Field(ge=0, default=0, le=2)
    ca: int = Field(ge=0, default=0, le=3)
    thal: int = Field(ge=0, default=0, le=2)


@app.post("/predict")
def predict(patient: Patient):
    patient_dict = patient.model_dump()
    df = pd.DataFrame([patient_dict], columns=FEATURE_COLUMNS)

    #chol_log
    df['chol_log'] = np.log1p(df['chol'])

    #hr_reserve_ratio
    df['max_hr_expected'] = 220 - df['age']
    df['hr_reserve_ratio'] = df['thalach'] / df['max_hr_expected']
    df = df.drop("max_hr_expected", axis=1)

    #chol_per_age
    df['chol_per_age'] = df['chol'] / df['age']

    #oldpeak_is_zero
    df['oldpeak_is_zero'] = (df['oldpeak'] == 0).astype(int)


    y_proba = loaded_model.predict_proba(df)
    if y_proba[0][1] <= 0.40:
        result = "Вероятно пациент здоров"
    else:
        result = "Вероятно у пациента проблемы с сердцем"

    shap_values = explainer.shap_values(df)  # shape: (1, 17)
    patient_shap = shap_values[0]  # массив из 17 чисел
    shap_dict = dict(zip(df.columns, patient_shap))
    risk_factors = sorted(
        [(k, v) for k, v in shap_dict.items() if v > 0],
        key=lambda x: x[1], reverse=True
    )[:5]
    protective_factors = sorted(
        [(k, v) for k, v in shap_dict.items() if v < 0],
        key=lambda x: x[1]
    )[:5]


    return {
        "probability": float(y_proba[0][1]),
        "result": result,
        "risk_factors": [{"feature": k, "impact": float(v)} for k, v in risk_factors],
        "protective_factors": [{"feature": k, "impact": float(v)} for k, v in protective_factors]
    }


if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)

"""Пример пациента
{
  "age": 64,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 212,
  "fbs": 0,
  "restecg": 2,
  "thalach": 132,
  "exang": 0,
  "oldpeak": 2,
  "slope": 1,
  "ca": 2,
  "thal": 1
}
"""