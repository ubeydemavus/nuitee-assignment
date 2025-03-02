"""
Unit tests for the UnionFind data structure
"""
import pytest
from app.services.matching_service import UnionFind


def test_unionfind_init():
    """Test UnionFind initialization"""
    uf = UnionFind()
    assert uf.parent == {}


def test_unionfind_find_new_word():
    """Test find method with a new word"""
    uf = UnionFind()
    result = uf.find('test')
    assert result == 'test'
    assert uf.parent['test'] == 'test'


def test_unionfind_find_existing_word():
    """Test find method with an existing word"""
    uf = UnionFind()
    uf.parent = {'test': 'test', 'example': 'test'}
    result = uf.find('example')
    assert result == 'test'


def test_unionfind_find_with_path_compression():
    """Test find method with path compression"""
    uf = UnionFind()
    # Create a chain of parent relationships
    uf.parent = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'e'}
    result = uf.find('a')
    assert result == 'e'
    # Check if path was compressed
    assert uf.parent['a'] == 'e'
    assert uf.parent['b'] == 'e'
    assert uf.parent['c'] == 'e'
    assert uf.parent['d'] == 'e'


def test_unionfind_union():
    """Test union method"""
    uf = UnionFind()
    uf.union('word1', 'word2')
    uf.union('word2', 'word3')
    
    # Check if all words are in the same set
    assert uf.find('word1') == uf.find('word2')
    assert uf.find('word2') == uf.find('word3')
    assert uf.find('word1') == uf.find('word3')


def test_unionfind_union_with_existing_words():
    """Test union method with words in different sets"""
    uf = UnionFind()
    
    # Create two separate sets
    uf.union('a', 'b')
    uf.union('c', 'd')
    
    # The roots should be different
    assert uf.find('a') != uf.find('c')
    
    # Merge the sets
    uf.union('b', 'c')
    
    # Now all words should be in the same set
    assert uf.find('a') == uf.find('c')
    assert uf.find('b') == uf.find('d') 