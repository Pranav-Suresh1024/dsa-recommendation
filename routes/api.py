import os
from fastapi import APIRouter
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer
from pipeline.recommender.bkt import process_submission
from pydantic import BaseModel
from pipeline.recommender.hlr import process_hlr, calculate_urgency
from datetime import datetime, timezone
from pipeline.recommender.ranking import rank_candidates
import json
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
        submission.model_dump(),
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


# In memory HLR store — replace with database later
user_hlr_store = {}

@router.post("/update_hlr")
def update_hlr_endpoint(submission: Submission):
    current_hlr = user_hlr_store.get(submission.userId, {})
    
    updated_hlr, results = process_hlr(
        submission.model_dump(),
        current_hlr
    )
    
    user_hlr_store[submission.userId] = updated_hlr
    
    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "results": results,
        "updated_hlr": updated_hlr
    }

@router.get("/urgency/{user_id}")
def get_urgency(user_id: str):
    hlr_state = user_hlr_store.get(user_id, {})
    current_time = datetime.now(timezone.utc).timestamp()
    
    urgency_scores = {
        topic: calculate_urgency(state, current_time)
        for topic, state in hlr_state.items()
    }
    
    return {
        "userId": user_id,
        "urgency_scores": urgency_scores
    }


@router.get("/recommend/{user_id}")
def recommend(user_id: str, limit: int = 10):
    

    # Get user state
    mastery = user_mastery_store.get(user_id, {})
    hlr_state = user_hlr_store.get(user_id, {})
    current_time = datetime.now(timezone.utc).timestamp()

    # Score each topic by combining BKT and HLR
    topic_scores = {}
    all_topics = set(list(mastery.keys()) + list(hlr_state.keys()))

    for topic in all_topics:
        bkt_score = 1 - mastery.get(topic, 0.15)      # low mastery = high priority
        urgency = calculate_urgency(hlr_state.get(topic, {}), current_time)
        topic_scores[topic] = 0.6 * bkt_score + 0.4 * urgency

    # Pick top 3 weak/overdue topics
    top_topics = sorted(topic_scores, key=topic_scores.get, reverse=True)[:3]

    if not top_topics:
        return {"message": "No history found for user. Start solving problems first.", "candidates": []}

    # Query vector pool for each topic separately
    seen = set()
    candidates = []
    per_topic = max(1, limit // len(top_topics))

    for topic in top_topics:
        query_vector = model.encode(topic).tolist()
        results = client.query_points(
            collection_name="problems_v2",
            query=query_vector,
            limit=per_topic
        ).points

        for r in results:
            slug = r.payload["title_slug"]
            if slug not in seen:
                seen.add(slug)
                candidates.append({
                    "title_slug": slug,
                    "title": r.payload["title"],
                    "description": r.payload["description"],
                    "topics": r.payload["topics"],
                    "difficulty_score": r.payload["difficulty_score"],
                    "score": r.score
                })

    # Load topic topic edges for prerequisite check
    BASE_DIR_DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(BASE_DIR_DATA, "data", "topic_topic_edges_normalized.json"), encoding="utf-8-sig") as f:
        tt_edges = json.load(f)

    # Rank candidates
    ranked = rank_candidates(
        candidates=candidates,
        user_bkt_mastery=mastery,
        user_hlr_state=hlr_state,
        recent_topics=list(all_topics)[-10:],
        topic_topic_edges=tt_edges,
        current_timestamp=current_time
    )

    return {
        "userId": user_id,
        "top_topics": top_topics,
        "recommended": ranked[0] if ranked else None,
        "all_candidates_ranked": ranked
    }