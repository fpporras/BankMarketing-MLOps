from typing import Optional

from pydantic import BaseModel, Field


class BankMarketingRequest(BaseModel):

    age: int = Field(..., ge=18, le=100)

    job: str

    marital: str

    education: str

    default: str

    balance: float

    housing: str

    loan: str

    contact: str

    day_of_week: int

    month: str

    duration: int

    campaign: int

    pdays: int

    previous: int

    poutcome: str


class PredictionResponse(BaseModel):

    prediction: int

    probability: Optional[float]

    model_version: str