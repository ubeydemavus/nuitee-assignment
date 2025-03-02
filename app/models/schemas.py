"""
Pydantic models for request and response schemas
"""
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

# Request Models
class ReferenceRoom(BaseModel):
    hotel_id: Union[str, int] = Field(..., description="Hotel identifier")
    lp_id: str = Field(..., description="LP identifier")
    room_id: Union[str, int] = Field(..., description="Room identifier")
    room_name: str = Field(..., description="Room name")

class InputRoom(BaseModel):
    hotel_id: Union[str, int] = Field(..., description="Hotel identifier")
    lp_id: str = Field(..., description="LP identifier")
    supplier_room_id: Union[str, int] = Field(..., description="Supplier room identifier")
    supplier_name: str = Field(..., description="Supplier name")
    supplier_room_name: str = Field(..., description="Supplier room name")

class RoomMatchRequest(BaseModel):
    referenceCatalog: List[ReferenceRoom] = Field(..., description="List of reference rooms")
    inputCatalog: List[InputRoom] = Field(..., description="List of input rooms to match")

# Response Models
class NERResult(BaseModel):
    roomType: List[List[Union[str, float]]] = Field(default_factory=list)
    classType: List[List[Union[str, float]]] = Field(default_factory=list)
    bedCount: List[List[Union[str, float]]] = Field(default_factory=list)
    view: List[List[Union[str, float]]] = Field(default_factory=list)
    bedType: List[List[Union[str, float]]] = Field(default_factory=list)
    features: List[List[Union[str, float]]] = Field(default_factory=list)

class MappedRoom(BaseModel):
    core_room_id: Optional[int] = None
    core_hotel_id: Union[str, int]
    lp_id: str
    supplier_room_id: Union[str, int]
    supplier_name: str
    supplier_room_name: str
    ners: Optional[Dict[str, List[tuple]]] = None
    reference_match_score: float

class MatchResult(BaseModel):
    room_name: str
    room_id: Union[str, int]
    hotel_id: Union[str, int]
    lp_id: str
    mapped_rooms: List[MappedRoom]

class UnmappedRoom(BaseModel):
    core_room_id: Optional[int] = None
    core_hotel_id: Union[str, int]
    lp_id: str
    supplier_room_id: Union[str, int]
    supplier_name: str
    supplier_room_name: str
    ners: Optional[Dict[str, List[tuple]]] = None

class RoomMatchResponse(BaseModel):
    results: List[MatchResult] = Field(default_factory=list)
    unmapped_rooms: List[UnmappedRoom] = Field(default_factory=list) 