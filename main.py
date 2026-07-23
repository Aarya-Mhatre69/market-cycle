"""
Production-ready FastAPI server exposing the Shankh Financial Advisor.
"""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.shankh.agents.financial_advisor import FinancialAdvisor
from src.shankh.utils import extract_text_content

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

advisor: FinancialAdvisor | None = None


# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global advisor

    try:
        advisor = FinancialAdvisor()
        logger.info("Financial Advisor initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize Financial Advisor.")
        raise

    yield

    logger.info("Shutting down Financial Advisor server.")


app = FastAPI(
    title="Shankh Financial Advisor API",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    "https://shankh-finagent.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class AdvisorRequest(BaseModel):
    question: str
    thread_id: str = "default"


class AdvisorResponse(BaseModel):
    response: str


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Shankh Financial Advisor",
    }


@app.post("/ask", response_model=AdvisorResponse)
async def ask_financial_advisor(request: AdvisorRequest):
    if advisor is None:
        raise HTTPException(
            status_code=503,
            detail="Financial Advisor not initialized.",
        )

    logger.info(
        "Processing query on thread '%s': %s",
        request.thread_id,
        request.question,
    )

    try:
        result = advisor.ask(
            request.question,
            thread_id=request.thread_id,
        )

        return AdvisorResponse(
            response=extract_text_content(result)
        )

    except Exception as e:
        logger.exception("Agent invocation failed.")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/")
async def root():
    return {
        "message": "Shankh Financial Advisor API",
        "docs": "/docs",
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # change if filename differs
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )