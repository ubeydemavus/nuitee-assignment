"""
Example client for the Nuitee Room Matching API
"""
import requests
import json

def main():
    """Run the example client"""
    
    # Example payload from the Example notepad
    url = "http://localhost:8000/room_match"
    
    payload = {
        "referenceCatalog": [
            {
                'hotel_id': 000000,
                'lp_id': 'lp000000',
                'room_id': 000000,
                'room_name': 'Deluxe Double Room'
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
            },
            {
                'hotel_id': 13876027,
                'lp_id': 'lp655653d7',
                'room_id': 1143508664,
                'room_name': 'Apartment'
            }
        ],
        "inputCatalog": [
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
            },
            {
                'hotel_id': 1679478,
                'lp_id': 'lp19a076',
                'supplier_room_id': 220832800,
                'supplier_name': 'Expedia',
                'supplier_room_name': 'Apartment, 2 Bedrooms'
            },
            {
                'hotel_id': 555940,
                'lp_id': 'lp87ba4',
                'supplier_room_id': 217865265,
                'supplier_name': 'Expedia',
                'supplier_room_name': 'Junior Suite'
            },
            {
                'hotel_id': 1701562941,
                'lp_id': 'lp656bca3d',
                'supplier_room_id': 324211309,
                'supplier_name': 'Expedia',
                'supplier_room_name': 'Deluxe Double Room'
            },
            {
                'hotel_id': 2180480,
                'lp_id': 'lp214580',
                'supplier_room_id': 214544056,
                'supplier_name': 'Expedia',
                'supplier_room_name': 'Apartment, 2 Bedrooms'
            },
            {
                'hotel_id': 1700326908,
                'lp_id': 'lp6558edfc',
                'supplier_room_id': 321315885,
                'supplier_name': 'Expedia',
                'supplier_room_name': 'Apartment'
            },
            {
                'hotel_id': 1736091,
                'lp_id': 'lp1a7d9b',
                'supplier_room_id': 220535440,
                'supplier_name': 'Expedia',
                'supplier_room_name': 'Apartment'
            },
            {
                'hotel_id': 463557,
                'lp_id': 'lp712c5',
                'supplier_room_id': 216630194,
                'supplier_name': 'Expedia',
                'supplier_room_name': 'Standard Double Room'
            }
        ]
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    print("Sending request to Room Matching API...")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("Request successful!")
        print("\nResponse:")
        pretty_response = json.dumps(response.json(), indent=2)
        print(pretty_response)
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main() 