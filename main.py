"""
Main FastAPI application file for Room Matching API
"""
import uvicorn
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.services.matching_service import RoomMatcher

# Initialize the room matcher service
room_matcher = RoomMatcher()

app = FastAPI(
    title="Room Matching API",
    description="API for matching hotel rooms based on names and attributes",
    version="1.0.0",
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Add a simple health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    # Check if data was loaded successfully
    data_loaded = hasattr(room_matcher, 'ref_df') and not room_matcher.ref_df.empty
    
    return {
        "status": "ok",
        "api_version": "1.0.0",
        "data_loaded": data_loaded,
        "datasets": {
            "reference_rooms": len(room_matcher.ref_d) if hasattr(room_matcher, 'ref_d') else 0,
            "core_rooms": len(room_matcher.core_d) if hasattr(room_matcher, 'core_d') else 0,
            "synonyms": len(room_matcher.synonym_dict) if hasattr(room_matcher, 'synonym_dict') else 0,
            "embeddings": len(room_matcher.embeds) if hasattr(room_matcher, 'embeds') else 0
        }
    }

# Add a shortcut for the room matching end point.
@app.post("/room_match", tags=["Room Matching"])
async def room_match_endpoint(request_data: dict):
    """
    This is a shortcut to the actual implementation in the API router.
    """
    from app.api.routes import room_match_dict
    return await room_match_dict(request_data, room_matcher=room_matcher)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 