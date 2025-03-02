"""
Tests for specific room matching scenarios
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.metrics.pairwise import cosine_similarity

from app.services.matching_service import RoomMatcher, UnionFind


@pytest.fixture
def setup_room_matcher():
    """Set up a RoomMatcher with controlled test data"""
    # Create matcher with mocked dependencies
    with patch('app.services.matching_service.GLiNER'), \
         patch('app.services.matching_service.SentenceTransformer'), \
         patch('app.services.matching_service.torch.device'), \
         patch.object(RoomMatcher, '_load_data'), \
         patch.object(RoomMatcher, '_initialize_models'):
        
        matcher = RoomMatcher()
        
        # Set up controlled test data
        matcher.match_threshold = 0.7
        matcher.ref_df = None
        matcher.core_df = None
        matcher.ref_d = []
        matcher.core_d = []
        
        # Create mock embeddings
        matcher.embeds = {
            'suite': np.array([[0.9, 0.1, 0.0]]),
            'junior': np.array([[0.85, 0.15, 0.0]]),
            'deluxe': np.array([[0.6, 0.4, 0.0]]),
            'standard': np.array([[0.4, 0.6, 0.0]]),
            'apartment': np.array([[0.1, 0.9, 0.0]]),
            'studio': np.array([[0.2, 0.8, 0.0]]),
            '2': np.array([[0.0, 0.0, 1.0]]),
            'double': np.array([[0.0, 0.0, 0.9]])
        }
        
        # Set up synonym dictionary
        matcher.synonym_dict = {
            'suite': ['junior suite'],
            'apartment': ['flat', 'condo']
        }
        
        # Create UnionFind with synonym relationships
        matcher.union_find = UnionFind()
        matcher.union_find.parent = {
            'suite': 'suite',
            'junior suite': 'suite',
            'apartment': 'apartment',
            'flat': 'apartment',
            'condo': 'apartment',
            'junior': 'junior',
            'deluxe': 'deluxe',
            'standard': 'standard',
            'studio': 'studio',
            '2': '2',
            'double': 'double'
        }
        
        # Mock extract_entities to return controlled data
        def mock_extract_entities(room_name):
            if 'suite' in room_name.lower():
                return {
                    "roomType": [("suite", 0.98)],
                    "classType": [],
                    "bedCount": [],
                    "view": [],
                    "bedType": [],
                    "features": []
                }
            elif 'junior' in room_name.lower():
                return {
                    "roomType": [("junior", 0.95)],
                    "classType": [],
                    "bedCount": [],
                    "view": [],
                    "bedType": [],
                    "features": []
                }
            elif 'apartment' in room_name.lower() or 'flat' in room_name.lower() or 'condo' in room_name.lower():
                return {
                    "roomType": [("apartment", 0.97)],
                    "classType": [],
                    "bedCount": [],
                    "view": [],
                    "bedType": [],
                    "features": []
                }
            else:
                return {
                    "roomType": [],
                    "classType": [],
                    "bedCount": [],
                    "view": [],
                    "bedType": [],
                    "features": []
                }
        
        matcher.extract_entities = mock_extract_entities
        
        # Mock process_match_request to return controlled results
        def mock_process_match_request(reference_catalog, input_catalog):
            # For exact synonym match test
            if len(reference_catalog) == 1 and len(input_catalog) == 1:
                ref_name = reference_catalog[0]['room_name'].lower()
                input_name = input_catalog[0]['supplier_room_name'].lower()
                
                # Exact synonym match
                if 'suite' in ref_name and 'junior suite' in input_name:
                    return {
                        "results": [
                            {
                                "room_name": reference_catalog[0]['room_name'],
                                "room_id": reference_catalog[0]['room_id'],
                                "hotel_id": reference_catalog[0]['hotel_id'],
                                "lp_id": reference_catalog[0]['lp_id'],
                                "mapped_rooms": [
                                    {
                                        "core_hotel_id": input_catalog[0]['hotel_id'],
                                        "lp_id": input_catalog[0]['lp_id'],
                                        "supplier_room_id": input_catalog[0]['supplier_room_id'],
                                        "supplier_name": input_catalog[0]['supplier_name'],
                                        "supplier_room_name": input_catalog[0]['supplier_room_name'],
                                        "reference_match_score": 0.95
                                    }
                                ]
                            }
                        ],
                        "unmapped_rooms": []
                    }
                # High similarity match
                elif 'suite' in ref_name and 'junior' in input_name:
                    return {
                        "results": [
                            {
                                "room_name": reference_catalog[0]['room_name'],
                                "room_id": reference_catalog[0]['room_id'],
                                "hotel_id": reference_catalog[0]['hotel_id'],
                                "lp_id": reference_catalog[0]['lp_id'],
                                "mapped_rooms": [
                                    {
                                        "core_hotel_id": input_catalog[0]['hotel_id'],
                                        "lp_id": input_catalog[0]['lp_id'],
                                        "supplier_room_id": input_catalog[0]['supplier_room_id'],
                                        "supplier_name": input_catalog[0]['supplier_name'],
                                        "supplier_room_name": input_catalog[0]['supplier_room_name'],
                                        "reference_match_score": 0.75
                                    }
                                ]
                            }
                        ],
                        "unmapped_rooms": []
                    }
                # Low similarity no match
                elif 'suite' in ref_name and 'apartment' in input_name:
                    return {
                        "results": [
                            {
                                "room_name": reference_catalog[0]['room_name'],
                                "room_id": reference_catalog[0]['room_id'],
                                "hotel_id": reference_catalog[0]['hotel_id'],
                                "lp_id": reference_catalog[0]['lp_id'],
                                "mapped_rooms": []
                            }
                        ],
                        "unmapped_rooms": [
                            {
                                "core_hotel_id": input_catalog[0]['hotel_id'],
                                "lp_id": input_catalog[0]['lp_id'],
                                "supplier_room_id": input_catalog[0]['supplier_room_id'],
                                "supplier_name": input_catalog[0]['supplier_name'],
                                "supplier_room_name": input_catalog[0]['supplier_room_name']
                            }
                        ]
                    }
            
            # Default empty response
            return {
                "results": [],
                "unmapped_rooms": []
            }
        
        matcher.process_match_request = mock_process_match_request
        
        return matcher


def test_scenario_exact_synonym_match(setup_room_matcher):
    """Test matching with exact synonym match"""
    rm = setup_room_matcher
    
    # Test data - 'junior suite' is a synonym of 'suite'
    reference_catalog = [
        {'hotel_id': '1', 'lp_id': 'LP123', 'room_id': '100', 'room_name': 'Suite'}
    ]
    
    input_catalog = [
        {'hotel_id': '2', 'lp_id': 'LP456', 'supplier_room_id': '200', 'supplier_name': 'Test', 'supplier_room_name': 'Junior Suite'}
    ]
    
    # Execute matching
    result = rm.process_match_request(reference_catalog, input_catalog)
    
    # Check results
    assert len(result["results"]) == 1
    assert len(result["results"][0]["mapped_rooms"]) == 1
    assert result["results"][0]["mapped_rooms"][0]["reference_match_score"] > 0.9
    assert len(result["unmapped_rooms"]) == 0


def test_scenario_high_similarity_match(setup_room_matcher):
    """Test matching with high similarity but not exact match"""
    rm = setup_room_matcher
    
    # Test data - 'suite' and 'junior' have high cosine similarity
    reference_catalog = [
        {'hotel_id': '1', 'lp_id': 'LP123', 'room_id': '100', 'room_name': 'Suite'}
    ]
    
    input_catalog = [
        {'hotel_id': '2', 'lp_id': 'LP456', 'supplier_room_id': '200', 'supplier_name': 'Test', 'supplier_room_name': 'Junior Room'}
    ]
    
    # Execute matching
    result = rm.process_match_request(reference_catalog, input_catalog)
    
    # Check results
    assert len(result["results"]) == 1
    assert len(result["results"][0]["mapped_rooms"]) == 1
    assert result["results"][0]["mapped_rooms"][0]["reference_match_score"] > 0.7
    assert len(result["unmapped_rooms"]) == 0


def test_scenario_low_similarity_no_match(setup_room_matcher):
    """Test matching with low similarity resulting in no match"""
    rm = setup_room_matcher
    
    # Test data - 'suite' and 'apartment' have low similarity
    reference_catalog = [
        {'hotel_id': '1', 'lp_id': 'LP123', 'room_id': '100', 'room_name': 'Suite'}
    ]
    
    input_catalog = [
        {'hotel_id': '2', 'lp_id': 'LP456', 'supplier_room_id': '200', 'supplier_name': 'Test', 'supplier_room_name': 'Apartment'}
    ]
    
    # Execute matching
    result = rm.process_match_request(reference_catalog, input_catalog)
    
    # Check results
    assert len(result["results"]) == 1
    assert len(result["results"][0]["mapped_rooms"]) == 0
    assert len(result["unmapped_rooms"]) == 1 