"""
Integration tests for the client example
"""
import pytest
from unittest.mock import patch, MagicMock

import client_example


@patch('client_example.requests.post')
def test_client_example_success(mock_post):
    """Test successful API call in client example"""
    # Setup mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
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
        "unmapped_rooms": []
    }
    mock_post.return_value = mock_response
    
    # Call client
    with patch('builtins.print') as mock_print:
        client_example.main()
    
    # Verify post was called with correct parameters
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "room_match" in args[0]  # URL is the first positional argument
    assert "json" in kwargs
    assert "headers" in kwargs
    assert kwargs["headers"]["content-type"] == "application/json"
    
    # Verify response was processed correctly
    mock_response.json.assert_called_once()
    mock_print.assert_called()


@patch('client_example.requests.post')
def test_client_example_error(mock_post):
    """Test API error handling in client example"""
    # Setup mock
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response
    
    # Call client
    with patch('builtins.print') as mock_print:
        client_example.main()
    
    # Verify response was processed correctly
    mock_print.assert_any_call("Request failed with status code: 500")
    mock_print.assert_any_call("Internal Server Error") 