import os
from fastapi import APIRouter
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer
from pipeline.recommender.bkt import process_submission
from pydantic import BaseModel

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = QdrantClient(path=os.path.join(BASE_DIR, "qdrant_storage_v2"))
model = SentenceTransformer('all-MiniLM-L6-v2')

user_mastery_store = {}

class Submission(BaseModel):
    userId: str
    problemId: str
    verdict: str
    testCasesPassed: int
    totalTestCases: int
    hintsUsed: int
    submissionCount: int
    normalisedScore: float
    timestamp: float

@router.get("/candidates")
def get_candidates(topic: str, min_difficulty: int = 1, max_difficulty: int = 3, limit: int = 10):
    query_vector = model.encode(topic).tolist()
    results = client.query_points(
        collection_name="problems_v2",
        query=query_vector,
        limit=limit,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="difficulty_score",
                    range=Range(gte=min_difficulty, lte=max_difficulty)
                )
            ]
        )
    ).points
    return [
        {
            "title_slug": r.payload["title_slug"],
            "title": r.payload["title"],
            "description": r.payload["description"],
            "topics": r.payload["topics"],
            "difficulty_score": r.payload["difficulty_score"],
            "score": r.score
        }
        for r in results
    ]

@router.post("/update_bkt")
def update_bkt_endpoint(submission: Submission):
    current_mastery = user_mastery_store.get(submission.userId, {})
    updated_mastery, mastered_topics, results = process_submission(
        submission.dict(),
        current_mastery
    )
    user_mastery_store[submission.userId] = updated_mastery
    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "results": results,
        "mastered_topics": mastered_topics,
        "updated_mastery": updated_mastery
    }

@router.get("/mastery/{user_id}")
def get_mastery(user_id: str):
    mastery = user_mastery_store.get(user_id, {})
    return {
        "userId": user_id,
        "mastery": mastery,
        "mastered_topics": [t for t, v in mastery.items() if v >= 0.75]
    }