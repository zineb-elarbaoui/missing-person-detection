from fastapi import FastAPI

app = FastAPI()  # Ensure this line is present at the global level

@app.get("/")
def read_root():
    return {"message": "Missing Person Detection API is running."}
