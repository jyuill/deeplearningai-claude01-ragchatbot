"""
Shared test fixtures and configuration for RAG chatbot testing
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk
from search_tools import CourseSearchTool, CourseOutlineTool
from ai_generator import AIGenerator

@pytest.fixture
def test_config():
    """Create a test configuration"""
    config = Config()
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.ANTHROPIC_API_KEY = "test-api-key"
    return config

@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction to Vectors", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Embedding Basics", lesson_link="https://example.com/lesson2"),
        Lesson(lesson_number=3, title="Similarity Search", lesson_link="https://example.com/lesson3")
    ]
    
    return Course(
        title="Advanced Retrieval for AI with Chroma",
        instructor="John Doe",
        course_link="https://example.com/course",
        lessons=lessons
    )

@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = [
        CourseChunk(
            content="Vector databases are essential for modern AI applications. They enable semantic search.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Embeddings convert text into numerical vectors that capture semantic meaning.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Similarity search finds the most relevant documents based on vector distance.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]
    return chunks

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for isolated testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Default search results
    mock_results = SearchResults(
        documents=["Sample document content about vectors"],
        metadata=[{"course_title": "Advanced Retrieval for AI with Chroma", "lesson_number": 1}],
        distances=[0.1],
        error=None
    )
    mock_store.search.return_value = mock_results
    
    # Course name resolution
    mock_store._resolve_course_name.return_value = "Advanced Retrieval for AI with Chroma"
    
    # Course metadata
    mock_store.get_all_courses_metadata.return_value = [{
        "title": "Advanced Retrieval for AI with Chroma",
        "instructor": "John Doe",
        "course_link": "https://example.com/course",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction to Vectors", "lesson_link": "https://example.com/lesson1"},
            {"lesson_number": 2, "lesson_title": "Embedding Basics", "lesson_link": "https://example.com/lesson2"}
        ]
    }]
    
    # Lesson link retrieval
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    
    return mock_store

@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for AI generator testing"""
    mock_client = Mock()
    
    # Mock successful response without tool use
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response")]
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_tool_use_response():
    """Create a mock Anthropic response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "test query"}
    
    mock_response.content = [mock_tool_block]
    
    return mock_response

@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "Vector databases store embeddings for semantic search",
            "ChromaDB is a popular vector database solution"
        ],
        metadata=[
            {"course_title": "Advanced Retrieval for AI with Chroma", "lesson_number": 1},
            {"course_title": "Advanced Retrieval for AI with Chroma", "lesson_number": 2}
        ],
        distances=[0.1, 0.2],
        error=None
    )

@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )

@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Test search error")