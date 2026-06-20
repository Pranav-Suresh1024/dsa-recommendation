from fastapi import FastAPI
from routes.api import router

app = FastAPI()
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to Recommendation Service"}