"""
DraftWalk Backend – FastAPI + OpenCV floor-plan processing.

Endpoints:
  POST /api/process   – Upload floor-plan image -> JSON SceneGraph
  POST /api/prompt    – Send AI prompt + scene -> JSON Patch
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import uuid

from processing import extract_scene_from_image

app = FastAPI(title="DraftWalk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    scene: dict[str, Any]

class PromptResponse(BaseModel):
    message: str
    patch: dict[str, Any] | None = None

@app.post("/api/process")
async def process_floor_plan(file: UploadFile = File(...)):
    contents = await file.read()
    scene = extract_scene_from_image(contents)
    return scene

@app.post("/api/prompt", response_model=PromptResponse)
async def handle_prompt(req: PromptRequest):
    """
    Simulated LLM response for natural language prompting.
    """
    prompt_lower = req.prompt.lower()
    patch: dict[str, Any] | None = None
    message = ""

    if "color" in prompt_lower or "colour" in prompt_lower or "make" in prompt_lower:
        color_map = {
            "red": "#c0392b", "blue": "#3a7bd5", "green": "#27ae60",
            "white": "#ffffff", "black": "#2c2825", "yellow": "#f1c40f",
            "brown": "#8b6f47", "pink": "#e91e63", "purple": "#8e44ad",
        }
        found_color = None
        for name, hex_val in color_map.items():
            if name in prompt_lower:
                found_color = hex_val
                break

        if found_color:
            updated_objects = [{**obj, "color": found_color} for obj in req.scene.get("objects", [])]
            patch = {"objects": updated_objects}
            message = f"Updated object colors to {found_color}."
        else:
            message = "I couldn't identify the color. Try e.g., 'make the sofa blue'."

    elif "add" in prompt_lower and ("sofa" in prompt_lower or "couch" in prompt_lower):
        new_obj = {
            "id": f"obj_{uuid.uuid4().hex[:8]}",
            "type": "furniture", "label": "New Sofa",
            "position": {"x": 3, "y": 0.4, "z": 6}, "rotation": {"x": 0, "y": 0, "z": 0},
            "scale": {"x": 2, "y": 0.8, "z": 0.9}, "color": "#6b8f71", "geometry": "box",
        }
        patch = {"objects": req.scene.get("objects", []) + [new_obj]}
        message = "Added a sofa."

    elif "remove" in prompt_lower or "delete" in prompt_lower:
        existing = req.scene.get("objects", [])
        if existing:
            patch = {"objects": existing[:-1]}
            message = f"Removed last object."
        else:
            message = "No objects to remove."

    else:
        message = "Demo AI mode: Try 'make objects blue', 'add a sofa', or 'remove object'."

    return PromptResponse(message=message, patch=patch)
