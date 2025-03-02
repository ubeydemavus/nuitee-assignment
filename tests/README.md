# Room Matching API Tests

This directory contains tests for the Room Matching API.

## Test Structure

- `unit/`: Unit tests for individual components
- `integration/`: Integration tests for API endpoints and interactions
- `data/`: Test data files

## Running Tests

### Install development dependencies

```bash
pip install -r requirements-dev.txt
```

### Run all tests with coverage

```bash
python run_tests.py
```

Or:

```bash
pytest --cov=app --cov=main --cov-report=term --cov-report=html
```

### Run specific test categories

```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run a specific test file
pytest tests/unit/test_unionfind.py
```

## Coverage Reports

Running the tests with the coverage options will generate:

- A terminal report showing coverage percentages
- An HTML report in the `htmlcov/` directory

To view the HTML report, open `htmlcov/index.html` in a web browser.

## Test Fixtures

Common test fixtures are defined in `conftest.py`. These include:

- `test_client`: A FastAPI test client
- `sample_reference_rooms`: Sample reference room data for testing
- `sample_input_rooms`: Sample input room data for testing
- `sample_request_payload`: Sample API request payload
- `mock_room_matcher`: A mocked RoomMatcher instance

## Creating New Tests

When creating new tests, follow these guidelines:

1. Place unit tests in the `unit/` directory
2. Place integration tests in the `integration/` directory
3. Use the existing fixtures when possible
4. Mock external dependencies to make tests fast and reliable
5. Name test files with the `test_` prefix
6. Name test functions with the `test_` prefix 