"""
API routes for room matching
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional

from app.models.schemas import RoomMatchRequest, RoomMatchResponse
from app.services.matching_service import RoomMatcher

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Room Matching"])

# Dependency to get the RoomMatcher service
def get_room_matcher() -> RoomMatcher:
    """Dependency to get RoomMatcher service instance"""
    return RoomMatcher()

@router.post("/room_match", response_model=RoomMatchResponse)
async def room_match(
    request: RoomMatchRequest, 
    room_matcher: RoomMatcher = Depends(get_room_matcher)
) -> Dict[str, Any]:
    """
    Match rooms from input catalog to reference catalog.
    
    This endpoint takes a list of reference rooms and input rooms and returns matching results.
    """
    try:
        logger.info(f"Processing match request with {len(request.referenceCatalog)} reference rooms and {len(request.inputCatalog)} input rooms")
        
        # Convert Pydantic models to dictionaries for the service
        reference_catalog = [ref.dict() for ref in request.referenceCatalog]
        input_catalog = [inp.dict() for inp in request.inputCatalog]
        
        # Process the match request
        result = room_matcher.process_match_request(reference_catalog, input_catalog)
        
        logger.info(f"Match completed. Found {len(result['results'])} matches and {len(result['unmapped_rooms'])} unmapped rooms")
        return result
        
    except Exception as e:
        logger.exception("Error processing room match request")
        raise HTTPException(status_code=500, detail=f"Error processing match request: {str(e)}")

# This is for direct dict input (used by the shortcut in main.py)
async def room_match_dict(request_data: dict, room_matcher: Optional[RoomMatcher] = None) -> Dict[str, Any]:
    """
    Alternative entry point that accepts a dict directly.
    This is used by the shortcut in main.py.
    """
    try:
        # Convert to Pydantic model for validation
        request = RoomMatchRequest(**request_data)
        
        # Use the provided room_matcher or create a new one
        if room_matcher is None:
            room_matcher = RoomMatcher()
        
        # Convert Pydantic models to dictionaries for the service
        reference_catalog = [ref.dict() for ref in request.referenceCatalog]
        input_catalog = [inp.dict() for inp in request.inputCatalog]
        
        # Process the match request
        result = room_matcher.process_match_request(reference_catalog, input_catalog)
        
        return result
        
    except Exception as e:
        logger.exception("Error processing room match request")
        raise HTTPException(status_code=500, detail=f"Error processing match request: {str(e)}") 