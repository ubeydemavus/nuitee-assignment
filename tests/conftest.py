"""
Pytest configuration and fixtures for tests
"""
import pytest
import os
import pandas as pd
import pickle
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from main import app
from app.services.matching_service import RoomMatcher, UnionFind


@pytest.fixture
def test_client():
    """
    Create a test client for FastAPI app
    """
    return TestClient(app)


@pytest.fixture
def sample_reference_rooms():
    """
    Sample reference room catalog for testing
    """
    return [
        {
            'hotel_id': 13705497,
            'lp_id': 'lp182dfe',
            'room_id': 1142671139,
            'room_name': 'Suite'
        },
        {
            'hotel_id': 13556897,
            'lp_id': 'lp26db8f',
            'room_id': 1141987223,
            'room_name': 'Comfort Triple Room'
        },
        {
            'hotel_id': '000000',
            'lp_id': 'lp000000',
            'room_id': '0000000',
            'room_name': 'standart, studio, 2 beds'
        }
    ]


@pytest.fixture
def sample_input_rooms():
    """
    Sample input room catalog for testing
    """
    return [
        {
            'hotel_id': 719235,
            'lp_id': 'lpaf983',
            'supplier_room_id': 202312114,
            'supplier_name': 'Expedia',
            'supplier_room_name': 'Deluxe Twin Room'
        },
        {
            'hotel_id': 1700209184,
            'lp_id': 'lp65572220',
            'supplier_room_id': 314291539,
            'supplier_name': 'Expedia',
            'supplier_room_name': 'Apartment'
        },
        {
            'hotel_id': 203498,
            'lp_id': 'lp31aea',
            'supplier_room_id': 89974,
            'supplier_name': 'Expedia',
            'supplier_room_name': 'Junior Suite'
        }
    ]


@pytest.fixture
def sample_request_payload(sample_reference_rooms, sample_input_rooms):
    """
    Sample request payload for the API
    """
    return {
        "referenceCatalog": sample_reference_rooms,
        "inputCatalog": sample_input_rooms
    }


@pytest.fixture
def mock_room_matcher():
    """
    Mock room matcher with controlled behavior for testing
    """
    with patch('app.services.matching_service.RoomMatcher', autospec=True) as MockRoomMatcher:
        # Create a mock instance
        instance = MockRoomMatcher.return_value
        
        # Mock the process_match_request method
        instance.process_match_request.return_value = {
            "results": [
                {
                    "room_name": "Suite",
                    "room_id": 1142671139,
                    "hotel_id": 13705497,
                    "lp_id": "lp182dfe",
                    "mapped_rooms": [
                        {
                            "core_hotel_id": 1700209184,
                            "lp_id": "lp65572220",
                            "supplier_room_id": 314291539,
                            "supplier_name": "Expedia",
                            "supplier_room_name": "Apartment",
                            "reference_match_score": 0.85
                        }
                    ]
                }
            ],
            "unmapped_rooms": [
                {
                    "core_hotel_id": 719235,
                    "lp_id": "lpaf983",
                    "supplier_room_id": 202312114,
                    "supplier_name": "Expedia",
                    "supplier_room_name": "Deluxe Twin Room"
                },
                {
                    "core_hotel_id": 203498,
                    "lp_id": "lp31aea",
                    "supplier_room_id": 89974,
                    "supplier_name": "Expedia",
                    "supplier_room_name": "Junior Suite"
                }
            ]
        }
        
        yield instance


@pytest.fixture
def sample_ners_result():
    """
    Sample NER result for testing
    """
    return {
        "roomType": [("suite", 0.98)],
        "classType": [],
        "bedCount": [],
        "view": [],
        "bedType": [],
        "features": []
    }


@pytest.fixture
def sample_embeds():
    """
    Sample embeddings dictionary for testing
    """
    return {
        "suite": [[0.1, 0.2, 0.3]],
        "apartment": [[0.15, 0.25, 0.35]],
        "deluxe": [[0.2, 0.3, 0.4]]
    }


@pytest.fixture
def sample_synonym_dict():
    """
    Sample synonym dictionary for testing
    """
    return {
        "suite": ["junior suite", "executive suite"],
        "apartment": ["flat", "condo"],
        "studio": ["studio apartment"]
    } 