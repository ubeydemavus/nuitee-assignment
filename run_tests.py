#!/usr/bin/env python
"""
Test runner script for the Room Matching API
"""
import sys
import pytest


def main():
    """Run tests with specified arguments"""
    args = sys.argv[1:]
    
    # Default to running all tests if no args provided
    if not args:
        args = ["tests"]
    
    # Add coverage options if not specified
    coverage_options = [
        "--cov=app", 
        "--cov=main", 
        "--cov-report=term", 
        "--cov-report=html"
    ]
    
    has_coverage = any("--cov" in arg for arg in args)
    if not has_coverage:
        args.extend(coverage_options)
    
    # Run tests
    result = pytest.main(args)
    sys.exit(result)


if __name__ == "__main__":
    main() 