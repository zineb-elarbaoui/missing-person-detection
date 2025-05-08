from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from bson import ObjectId
from PIL import Image
import numpy as np
import torch
import io
from scipy.spatial.distance import cosine
import cv2

from models import PersonModel
from database import person_collection
from facenet_pytorch import InceptionResnetV1, MTCNN


app = FastAPI()

# Initialize FaceNet model
mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()


# --------------------- Person Endpoints ---------------------

# Get all persons
@app.get("/persons")
async def get_persons():
    persons = []
    async for person in person_collection.find():
        person["_id"] = str(person["_id"])
        persons.append(person)
    return persons


# Add a new person with face embedding
@app.post("/persons")
async def add_person(person: PersonModel, embedding: List[float]):
    if len(embedding) != 512:
        raise HTTPException(status_code=400, detail="Invalid embedding size")
    person_data = person.dict()
    person_data["embedding"] = embedding
    result = await person_collection.insert_one(person_data)
    return {"id": str(result.inserted_id)}


# Get person by ID
@app.get("/persons/{person_id}")
async def get_person(person_id: str):
    person = await person_collection.find_one({"_id": ObjectId(person_id)})
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    person["_id"] = str(person["_id"])
    return person


# Delete person by ID
@app.delete("/persons/{person_id}")
async def delete_person(person_id: str):
    result = await person_collection.delete_one({"_id": ObjectId(person_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Person not found")
    return {"message": "Person deleted successfully"}


# --------------------- Image Handling & Recognition ---------------------

# Upload image just to test
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return {"message": "Image received", "shape": img_array.shape}


# Helper: Extract face embedding from image bytes
def extract_embedding(image_bytes: bytes) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    aligned_faces = mtcnn(pil_image)

    if aligned_faces is None:
        raise HTTPException(status_code=400, detail="No face detected")

    if isinstance(aligned_faces, list):
        aligned_faces = aligned_faces[0]

    embedding_tensor = facenet(aligned_faces.unsqueeze(0))  # Add batch dim
    embedding = embedding_tensor.detach().cpu().numpy()[0]  # Convert to 1D array
    return embedding


# Recognize face from uploaded image
@app.post("/recognize_face/")
async def recognize_face(image: UploadFile = File(...)):
    image_bytes = await image.read()
    embedding = extract_embedding(image_bytes)

    persons = []
    async for person in person_collection.find():
        person["_id"] = str(person["_id"])
        persons.append(person)

    for person in persons:
        stored_embedding = np.array(person["embedding"])
        distance = cosine(embedding, stored_embedding)

        if distance < 0.6:
            return {
                "match": True,
                "person_id": person["_id"],
                "name": person.get("name", "Unknown"),
                "distance": distance
            }

    return {"match": False, "message": "No match found"}
