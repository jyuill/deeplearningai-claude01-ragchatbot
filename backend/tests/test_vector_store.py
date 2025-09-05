"""
Integration tests for VectorStore with real ChromaDB operations
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore:
    """Integration tests for VectorStore"""

    def test_init(self, temp_chroma_path, test_config):
        """Test VectorStore initialization"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL, max_results=5)
        
        assert store.max_results == 5
        assert store.client is not None
        assert store.course_catalog is not None
        assert store.course_content is not None

    def test_add_course_metadata(self, temp_chroma_path, test_config, sample_course):
        """Test adding course metadata to catalog"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add course metadata
        store.add_course_metadata(sample_course)
        
        # Verify course was added
        existing_titles = store.get_existing_course_titles()
        assert sample_course.title in existing_titles
        
        # Verify course count
        assert store.get_course_count() == 1

    def test_add_course_content(self, temp_chroma_path, test_config, sample_course_chunks):
        """Test adding course content chunks"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add content chunks
        store.add_course_content(sample_course_chunks)
        
        # Verify content was added by searching
        results = store.search("vector databases")
        
        # Should find results (exact match will depend on embedding model)
        assert isinstance(results, SearchResults)
        assert results.error is None

    def test_search_basic_functionality(self, temp_chroma_path, test_config, sample_course, sample_course_chunks):
        """Test basic search functionality"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL, max_results=3)
        
        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Search for content
        results = store.search("vector databases")
        
        # Verify results structure
        assert isinstance(results, SearchResults)
        assert results.error is None
        assert isinstance(results.documents, list)
        assert isinstance(results.metadata, list)
        assert isinstance(results.distances, list)
        assert len(results.documents) == len(results.metadata)
        assert len(results.documents) == len(results.distances)

    def test_search_with_course_name_filter(self, temp_chroma_path, test_config, sample_course, sample_course_chunks):
        """Test search with course name filtering"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Search with course filter
        results = store.search("embedding", course_name="Advanced Retrieval")
        
        # Verify results
        assert isinstance(results, SearchResults)
        assert results.error is None
        
        # All results should be from the specified course
        for metadata in results.metadata:
            assert metadata["course_title"] == sample_course.title

    def test_search_with_lesson_number_filter(self, temp_chroma_path, test_config, sample_course, sample_course_chunks):
        """Test search with lesson number filtering"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Search with lesson filter
        results = store.search("embedding", lesson_number=2)
        
        # Verify results
        assert isinstance(results, SearchResults)
        assert results.error is None
        
        # All results should be from the specified lesson
        for metadata in results.metadata:
            assert metadata.get("lesson_number") == 2

    def test_search_with_both_filters(self, temp_chroma_path, test_config, sample_course, sample_course_chunks):
        """Test search with both course and lesson filters"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Search with both filters
        results = store.search("similarity", course_name="Advanced Retrieval", lesson_number=3)
        
        # Verify results
        assert isinstance(results, SearchResults)
        assert results.error is None
        
        # Results should match both filters
        for metadata in results.metadata:
            assert metadata["course_title"] == sample_course.title
            assert metadata.get("lesson_number") == 3

    def test_search_nonexistent_course(self, temp_chroma_path, test_config, sample_course, sample_course_chunks):
        """Test search with nonexistent course filter"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Search with nonexistent course
        results = store.search("vectors", course_name="Nonexistent Course")
        
        # Should return error
        assert results.error is not None
        assert "No course found matching" in results.error

    def test_search_with_limit_parameter(self, temp_chroma_path, test_config, sample_course, sample_course_chunks):
        """Test search with custom limit parameter"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL, max_results=10)
        
        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Search with custom limit
        results = store.search("vector", limit=2)
        
        # Should respect the limit parameter
        assert len(results.documents) <= 2

    def test_resolve_course_name(self, temp_chroma_path, test_config, sample_course):
        """Test course name resolution functionality"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add course metadata
        store.add_course_metadata(sample_course)
        
        # Test exact match
        resolved = store._resolve_course_name(sample_course.title)
        assert resolved == sample_course.title
        
        # Test partial match
        resolved = store._resolve_course_name("Advanced Retrieval")
        assert resolved == sample_course.title
        
        # Test case insensitive
        resolved = store._resolve_course_name("advanced retrieval")
        assert resolved == sample_course.title

    def test_resolve_nonexistent_course_name(self, temp_chroma_path, test_config, sample_course):
        """Test course name resolution with nonexistent course"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add course metadata
        store.add_course_metadata(sample_course)
        
        # Test nonexistent course
        resolved = store._resolve_course_name("Completely Different Course")
        assert resolved is None

    def test_get_existing_course_titles(self, temp_chroma_path, test_config, sample_course):
        """Test getting existing course titles"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Initially empty
        titles = store.get_existing_course_titles()
        assert len(titles) == 0
        
        # Add course
        store.add_course_metadata(sample_course)
        
        # Should now contain the course
        titles = store.get_existing_course_titles()
        assert len(titles) == 1
        assert sample_course.title in titles

    def test_get_course_count(self, temp_chroma_path, test_config, sample_course):
        """Test getting course count"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Initially zero
        assert store.get_course_count() == 0
        
        # Add course
        store.add_course_metadata(sample_course)
        
        # Should be one
        assert store.get_course_count() == 1

    def test_get_all_courses_metadata(self, temp_chroma_path, test_config, sample_course):
        """Test getting all courses metadata"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add course
        store.add_course_metadata(sample_course)
        
        # Get metadata
        metadata_list = store.get_all_courses_metadata()
        
        assert len(metadata_list) == 1
        metadata = metadata_list[0]
        
        # Verify structure
        assert metadata["title"] == sample_course.title
        assert metadata["instructor"] == sample_course.instructor
        assert metadata["course_link"] == sample_course.course_link
        assert "lessons" in metadata
        assert len(metadata["lessons"]) == len(sample_course.lessons)
        
        # Verify lesson structure
        lesson = metadata["lessons"][0]
        assert "lesson_number" in lesson
        assert "lesson_title" in lesson
        assert "lesson_link" in lesson

    def test_get_course_link(self, temp_chroma_path, test_config, sample_course):
        """Test getting course link by title"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add course
        store.add_course_metadata(sample_course)
        
        # Get course link
        link = store.get_course_link(sample_course.title)
        assert link == sample_course.course_link
        
        # Test nonexistent course
        link = store.get_course_link("Nonexistent Course")
        assert link is None

    def test_get_lesson_link(self, temp_chroma_path, test_config, sample_course):
        """Test getting lesson link by course and lesson number"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add course
        store.add_course_metadata(sample_course)
        
        # Get lesson link
        link = store.get_lesson_link(sample_course.title, 1)
        assert link == sample_course.lessons[0].lesson_link
        
        # Test nonexistent lesson
        link = store.get_lesson_link(sample_course.title, 99)
        assert link is None
        
        # Test nonexistent course
        link = store.get_lesson_link("Nonexistent Course", 1)
        assert link is None

    def test_clear_all_data(self, temp_chroma_path, test_config, sample_course, sample_course_chunks):
        """Test clearing all data from collections"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Verify data exists
        assert store.get_course_count() > 0
        
        # Clear data
        store.clear_all_data()
        
        # Verify data is cleared
        assert store.get_course_count() == 0
        assert len(store.get_existing_course_titles()) == 0

    def test_search_results_from_chroma(self):
        """Test SearchResults.from_chroma class method"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'meta1': 'value1'}, {'meta2': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'meta1': 'value1'}, {'meta2': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_search_results_from_empty_chroma(self):
        """Test SearchResults.from_chroma with empty results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
        assert results.is_empty()

    def test_search_results_empty_method(self):
        """Test SearchResults.empty class method"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
        assert results.is_empty()

    def test_search_results_is_empty(self):
        """Test SearchResults.is_empty method"""
        # Empty results
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty()
        
        # Non-empty results
        non_empty_results = SearchResults(['doc1'], [{'meta': 'data'}], [0.1])
        assert not non_empty_results.is_empty()

    def test_build_filter_no_parameters(self, temp_chroma_path, test_config):
        """Test filter building with no parameters"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        filter_dict = store._build_filter(None, None)
        assert filter_dict is None

    def test_build_filter_course_only(self, temp_chroma_path, test_config):
        """Test filter building with course only"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        filter_dict = store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, temp_chroma_path, test_config):
        """Test filter building with lesson only"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        filter_dict = store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}

    def test_build_filter_both_parameters(self, temp_chroma_path, test_config):
        """Test filter building with both parameters"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        filter_dict = store._build_filter("Test Course", 1)
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 1}
        ]}
        assert filter_dict == expected

    @pytest.mark.slow
    def test_multiple_courses_search_isolation(self, temp_chroma_path, test_config):
        """Test that searches properly isolate between multiple courses"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Create two different courses
        course1 = Course(
            title="Python Basics",
            instructor="Alice",
            course_link="https://example.com/python",
            lessons=[Lesson(lesson_number=1, title="Variables", lesson_link="https://example.com/python/1")]
        )
        
        course2 = Course(
            title="JavaScript Fundamentals", 
            instructor="Bob",
            course_link="https://example.com/js",
            lessons=[Lesson(lesson_number=1, title="Functions", lesson_link="https://example.com/js/1")]
        )
        
        # Create chunks for each course
        chunks1 = [CourseChunk(content="Python variables store data", course_title=course1.title, lesson_number=1, chunk_index=0)]
        chunks2 = [CourseChunk(content="JavaScript functions are first-class", course_title=course2.title, lesson_number=1, chunk_index=0)]
        
        # Add both courses
        store.add_course_metadata(course1)
        store.add_course_metadata(course2)
        store.add_course_content(chunks1)
        store.add_course_content(chunks2)
        
        # Search should find both courses without filter
        results = store.search("functions")
        assert results.error is None
        
        # Search with Python filter should only find Python course
        results = store.search("variables", course_name="Python")
        assert results.error is None
        for metadata in results.metadata:
            assert metadata["course_title"] == course1.title

    def test_error_handling_in_search(self, temp_chroma_path, test_config):
        """Test error handling in search method"""
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL)
        
        # Mock the course_content collection to raise an exception
        with patch.object(store.course_content, 'query', side_effect=Exception("Test error")):
            results = store.search("test query")
            
            assert results.error is not None
            assert "Search error: Test error" in results.error
            assert results.is_empty()

    def test_max_results_configuration(self, temp_chroma_path, test_config, sample_course):
        """Test that max_results configuration is respected"""
        # Test with max_results = 2
        store = VectorStore(temp_chroma_path, test_config.EMBEDDING_MODEL, max_results=2)
        
        # Create multiple chunks to ensure we have more than 2 results
        chunks = []
        for i in range(5):
            chunk = CourseChunk(
                content=f"Vector content chunk {i} with similar semantic meaning",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=i
            )
            chunks.append(chunk)
        
        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(chunks)
        
        # Search should respect max_results
        results = store.search("vector")
        
        # Should get at most 2 results
        assert len(results.documents) <= 2