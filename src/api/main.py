# src/api/main.py
"""
The main FastAPI application for the AI Cooking Assistant.

This script creates the real-time API server that acts as the project's
Master Control Program (MCP). It defines all the endpoints required by the UI,
including:
- /query/assistant: For handling conversational AI queries.
- /recipes: For fetching and searching recipe data from the database.
- /vision/analyze: For processing real-time images from a camera feed.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

# --- Dynamic Path Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config_loader import get_config
from src.core.orchestrator import QueryOrchestrator
from src.models.sql_models import get_db_session, Recipe

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] [%(module)-20s] %(message)s')


# --- API Data Models (Pydantic) ---
# These models define the expected request and response structures for our API.

# For the main assistant endpoint
class QueryRequest(BaseModel):
    query_text: str
    user_id: Optional[str] = "default_user"


class QueryResponse(BaseModel):
    response_text: str
    intent: str
    source: str


# For the recipe fetching endpoint
class RecipeOut(BaseModel):
    title: str
    cuisine: Optional[str]
    ingredients: List[str]
    instructions: List[str]

    class Config:
        orm_mode = True  # Allows creating the model from a SQLAlchemy object


# For the vision analysis endpoint
class Detection(BaseModel):
    label: str
    confidence: float


class VisionResponse(BaseModel):
    detections: List[Detection]


# --- FastAPI Application Initialization ---
app = FastAPI(
    title="AI Cooking Assistant API",
    description="Real-time API to power the hands-free cooking companion.",
    version="1.0.0"
)


# --- Startup Event: Load Models and Orchestrator ---
# This function runs once when the server starts, ensuring heavy models
# are loaded into memory only one time.
@app.on_event("startup")
def startup_event():
    logging.info("--- Starting API Server and Loading Models ---")
    try:
        # Load the central configuration
        config = get_config()
        # Instantiate the main orchestrator and attach it to the app's state
        app.state.orchestrator = QueryOrchestrator(config)
        logging.info("✅ Orchestrator and core models loaded successfully.")
    except Exception as e:
        logging.exception(f"FATAL: Could not initialize the Query Orchestrator during startup: {e}")
        # This will prevent the app from starting if the core components fail to load.
        raise


# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "AI Cooking Assistant API is online"}


@app.post("/query/assistant", response_model=QueryResponse, tags=["AI Assistant"])
async def handle_query(request: QueryRequest):
    """
    Main endpoint to handle conversational user queries. It passes the query
    to the central orchestrator and returns the generated response.
    """
    if not request.query_text or not request.query_text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    logging.info(f"Received query from user '{request.user_id}': '{request.query_text}'")

    try:
        orchestrator: QueryOrchestrator = app.state.orchestrator
        response_data = orchestrator.handle_query(request.query_text)

        logging.info(
            f"Generated response from '{response_data['source']}': '{response_data['response_text'][:100]}...'")

        return QueryResponse(**response_data)
    except Exception as e:
        logging.exception(f"An error occurred while handling query: {request.query_text}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")


@app.get("/recipes", response_model=List[RecipeOut], tags=["Recipes"])
def get_all_recipes(cuisine: Optional[str] = None, search: Optional[str] = None):
    """
    Fetches all recipes from the database, with optional filters for
    cuisine and a search term for title.
    """
    db_session: Session = get_db_session(app.state.orchestrator.config.database.url)
    try:
        query = db_session.query(Recipe)
        if cuisine:
            query = query.filter(Recipe.cuisine.ilike(f"%{cuisine}%"))
        if search:
            query = query.filter(Recipe.title.ilike(f"%{search}%"))

        results = query.limit(100).all()
        return results
    except Exception as e:
        logging.exception("Failed to fetch recipes from the database.")
        raise HTTPException(status_code=500, detail="Could not retrieve recipes from the database.")
    finally:
        db_session.close()


@app.post("/vision/analyze", response_model=VisionResponse, tags=["Vision"])
async def analyze_image_from_ui(file: UploadFile = File(...)):
    """
    Receives an image from the UI, runs it through the YOLO model,
    and returns the detected objects. This is a conceptual endpoint.
    """
    # In a real implementation, the YOLO model would be loaded at startup
    # and attached to the app state, e.g., `app.state.yolo_model`.
    # The image bytes would be passed to it for inference.
    # image_bytes = await file.read()
    # detected_objects = app.state.yolo_model(image_bytes) ...

    # Returning mock data for UI demonstration purposes
    logging.info(f"Received image '{file.filename}' for vision analysis.")
    mock_detections = [
        {"label": "onion", "confidence": 0.92},
        {"label": "pan", "confidence": 0.88},
        {"label": "tomato", "confidence": 0.75},
        {"label": "knife", "confidence": 0.65},
    ]
    return VisionResponse(detections=mock_detections)

# --- To run this server locally: ---
# 1. Make sure all dependencies are installed from requirements.txt.
# 2. Ensure your .env file and config.yaml are correctly set up.
# 3. From your project's root directory, run the following command in your terminal:
#    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload