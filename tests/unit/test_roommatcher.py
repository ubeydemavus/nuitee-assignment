"""
Unit tests for the RoomMatcher class
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
import torch

from app.services.matching_service import RoomMatcher, UnionFind


@pytest.fixture
def mock_models():
    """Mock the NLP models to avoid actual loading"""
    with patch('app.services.matching_service.GLiNER') as mock_gliner, \
         patch('app.services.matching_service.SentenceTransformer') as mock_transformer, \
         patch('app.services.matching_service.torch.device', return_value='cpu'):
        
        # Mock GLiNER
        mock_gliner_instance = MagicMock()
        mock_gliner.from_pretrained.return_value = mock_gliner_instance
        mock_gliner_instance.to.return_value = mock_gliner_instance
        
        # Mock SentenceTransformer
        mock_transformer_instance = MagicMock()
        mock_transformer.return_value = mock_transformer_instance
        mock_transformer_instance.to.return_value = mock_transformer_instance
        
        yield {
            'gliner': mock_gliner_instance,
            'transformer': mock_transformer_instance
        }


@pytest.fixture
def mock_data_loading():
    """Mock data loading from files"""
    with patch('app.services.matching_service.pickle.load', return_value={}), \
         patch('app.services.matching_service.pd.read_csv', return_value=None), \
         patch('app.services.matching_service.os.path.join', return_value='mock_path'):
        yield


@pytest.fixture
def room_matcher(mock_models, mock_data_loading):
    """Create a RoomMatcher instance with mocked dependencies"""
    with patch.object(RoomMatcher, '_load_data'), \
         patch.object(RoomMatcher, 'build_synonym_groups'):
        matcher = RoomMatcher()
        matcher.ners_model = mock_models['gliner']
        matcher.embedding_model = mock_models['transformer']
        matcher.synonym_dict = {}
        matcher.embeds = {}
        matcher.ref_df = None
        matcher.core_df = None
        matcher.ref_d = []
        matcher.core_d = []
        # Initialize the union_find attribute directly since we're patching build_synonym_groups
        matcher.union_find = UnionFind()
        matcher.union_find.parent = {}
        return matcher


def test_room_matcher_init(room_matcher):
    """Test RoomMatcher initialization"""
    assert room_matcher.match_threshold == 0.1
    assert set(room_matcher.labels) == {"roomType", "classType", "bedCount", "view", "bedType", "features"}
    assert len(room_matcher.label_importance_coefficients) == 6


def test_get_final_score(room_matcher):
    """Test get_final_score method"""
    scores = {
        "roomType": 0.9,
        "classType": 0.8,
        "bedCount": 0.7,
        "view": 0.6,
        "bedType": 0.5,
        "features": 0.4
    }
    
    expected = (
        0.9 * (2/6) + 
        0.8 * (1/6) + 
        0.7 * (1/6) + 
        0.6 * (1/6) + 
        0.5 * (1/6) + 
        0.4 * (1/6)
    )
    
    result = room_matcher.get_final_score(scores)
    assert abs(result - expected) < 0.0001


def test_are_similar(room_matcher):
    """Test are_similar method"""
    # Set up mock data
    room_matcher.union_find.parent = {
        'suite': 'suite',
        'junior suite': 'suite',
        'apartment': 'apartment'
    }
    
    # Test similar words
    assert room_matcher.are_similar('suite', 'junior suite') == True
    
    # Test different words
    assert room_matcher.are_similar('suite', 'apartment') == False
    
    # Test unknown words
    assert room_matcher.are_similar('suite', 'unknown') == False
    assert room_matcher.are_similar('unknown', 'apartment') == False


@patch('app.services.matching_service.cosine_similarity')
def test_extract_entities(mock_cosine, room_matcher):
    """Test extract_entities method"""
    # Set up mocks
    room_matcher.ners_model.predict_entities.return_value = [
        {"label": "roomType", "text": "suite", "score": 0.98},
        {"label": "bedCount", "text": "2", "score": 0.95}
    ]
    
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    room_matcher.embedding_model.encode.return_value = embeddings
    
    # Call method
    result = room_matcher.extract_entities("Luxury Suite with 2 beds")
    
    # Check result
    assert "roomType" in result
    assert "bedCount" in result
    assert len(result["roomType"]) == 1
    assert result["roomType"][0] == ("suite", 0.98)
    assert len(result["bedCount"]) == 1
    assert result["bedCount"][0] == ("2", 0.95)
    
    # Check that other categories are empty
    assert len(result["classType"]) == 0
    assert len(result["view"]) == 0
    assert len(result["bedType"]) == 0
    assert len(result["features"]) == 0
    
    # Check that embeddings were updated
    assert "suite" in room_matcher.embeds
    assert "2" in room_matcher.embeds


@patch.object(RoomMatcher, 'process_match_request')
def test_process_match_request_direct_match(mock_process, room_matcher):
    """Test process_match_request with direct LP ID match"""
    # Setup mock return value
    expected_result = {
        "results": [
            {
                "room_name": "Suite",
                "room_id": "100",
                "hotel_id": "1",
                "lp_id": "LP123",
                "mapped_rooms": [
                    {
                        "core_hotel_id": "2",
                        "lp_id": "LP123",
                        "supplier_room_id": "200",
                        "supplier_name": "Test",
                        "supplier_room_name": "Deluxe Room",
                        "reference_match_score": 1.0
                    }
                ]
            }
        ],
        "unmapped_rooms": []
    }
    mock_process.return_value = expected_result
    
    # Test data
    reference_catalog = [
        {'hotel_id': '1', 'lp_id': 'LP123', 'room_id': '100', 'room_name': 'Suite'}
    ]
    
    input_catalog = [
        {'hotel_id': '2', 'lp_id': 'LP123', 'supplier_room_id': '200', 'supplier_name': 'Test', 'supplier_room_name': 'Deluxe Room'}
    ]
    
    # Call method
    result = room_matcher.process_match_request(reference_catalog, input_catalog)
    
    # Check results
    assert result == expected_result
    mock_process.assert_called_once_with(reference_catalog, input_catalog)


@patch.object(RoomMatcher, 'process_match_request')
def test_process_match_request_name_match(mock_process, room_matcher):
    """Test process_match_request with direct name match"""
    # Setup mock return value
    expected_result = {
        "results": [
            {
                "room_name": "Suite",
                "room_id": "100",
                "hotel_id": "1",
                "lp_id": "LP123",
                "mapped_rooms": [
                    {
                        "core_hotel_id": "2",
                        "lp_id": "LP456",
                        "supplier_room_id": "200",
                        "supplier_name": "Test",
                        "supplier_room_name": "Suite",
                        "reference_match_score": 1.0
                    }
                ]
            }
        ],
        "unmapped_rooms": []
    }
    mock_process.return_value = expected_result
    
    # Test data
    reference_catalog = [
        {'hotel_id': '1', 'lp_id': 'LP123', 'room_id': '100', 'room_name': 'Suite'}
    ]
    
    input_catalog = [
        {'hotel_id': '2', 'lp_id': 'LP456', 'supplier_room_id': '200', 'supplier_name': 'Test', 'supplier_room_name': 'Suite'}
    ]
    
    # Call method
    result = room_matcher.process_match_request(reference_catalog, input_catalog)
    
    # Check results
    assert result == expected_result
    mock_process.assert_called_once_with(reference_catalog, input_catalog) 