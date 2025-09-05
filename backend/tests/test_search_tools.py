"""
Unit tests for search tools (CourseSearchTool and CourseOutlineTool)
"""
import pytest
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        
        # Check all expected parameters are present
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

    def test_execute_successful_search(self, mock_vector_store, sample_search_results):
        """Test successful search execution"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("What are vector databases?")
        
        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="What are vector databases?",
            course_name=None,
            lesson_number=None
        )
        
        # Check formatted output
        assert "Advanced Retrieval for AI with Chroma - Lesson 1" in result
        assert "Vector databases store embeddings" in result
        assert "ChromaDB is a popular vector database" in result
        
        # Check that sources were tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Advanced Retrieval for AI with Chroma - Lesson 1"

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("embeddings", course_name="Advanced Retrieval")
        
        mock_vector_store.search.assert_called_once_with(
            query="embeddings",
            course_name="Advanced Retrieval",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("similarity search", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="similarity search",
            course_name=None,
            lesson_number=3
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test search with both course and lesson filters"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("vectors", course_name="Chroma", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="vectors",
            course_name="Chroma",
            lesson_number=1
        )

    def test_execute_with_search_error(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = error_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert result == "Test search error"

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("nonexistent content")
        
        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, mock_vector_store, empty_search_results):
        """Test empty results message includes filter information"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test", course_name="NonExistent", lesson_number=99)
        
        assert "No relevant content found" in result
        assert "in course 'NonExistent'" in result
        assert "in lesson 99" in result

    def test_source_tracking_with_lesson_links(self, mock_vector_store):
        """Test that lesson links are properly tracked in sources"""
        # Mock search results with lesson links
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Check that lesson link was requested
        mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 1)
        
        # Check that source includes URL
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"

    def test_source_tracking_without_lesson_number(self, mock_vector_store):
        """Test source tracking when no lesson number is present"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should not call get_lesson_link when no lesson number
        mock_vector_store.get_lesson_link.assert_not_called()
        
        # Source should not have URL
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["url"] is None


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["course_name"]
        
        # Check course_name parameter
        properties = definition["input_schema"]["properties"]
        assert "course_name" in properties

    def test_execute_successful_outline(self, mock_vector_store):
        """Test successful course outline retrieval"""
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Advanced Retrieval")
        
        # Verify course name resolution was called
        mock_vector_store._resolve_course_name.assert_called_once_with("Advanced Retrieval")
        
        # Verify metadata retrieval was called
        mock_vector_store.get_all_courses_metadata.assert_called_once()
        
        # Check formatted output contains expected elements
        assert "**Course:** Advanced Retrieval for AI with Chroma" in result
        assert "**Instructor:** John Doe" in result
        assert "**Course Link:** https://example.com/course" in result
        assert "**Course Outline (2 lessons):**" in result
        assert "Lesson 1: Introduction to Vectors" in result
        assert "Lesson 2: Embedding Basics" in result

    def test_execute_course_not_found(self, mock_vector_store):
        """Test handling when course is not found"""
        mock_vector_store._resolve_course_name.return_value = None
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("NonExistent Course")
        
        assert result == "No course found matching 'NonExistent Course'"

    def test_execute_metadata_not_found(self, mock_vector_store):
        """Test handling when course metadata is not found"""
        mock_vector_store.get_all_courses_metadata.return_value = []
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Advanced Retrieval")
        
        assert "Course metadata not found" in result

    def test_execute_course_without_lessons(self, mock_vector_store):
        """Test course outline with no lessons"""
        # Mock course with no lessons
        mock_vector_store.get_all_courses_metadata.return_value = [{
            "title": "Advanced Retrieval for AI with Chroma",
            "instructor": "John Doe",
            "course_link": "https://example.com/course",
            "lessons": []
        }]
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Advanced Retrieval")
        
        assert "**Course Outline:** No lessons available" in result

    def test_execute_course_with_lesson_links(self, mock_vector_store):
        """Test course outline includes lesson links when available"""
        # Course data already has lesson links in the mock
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Advanced Retrieval")
        
        # Check that lesson links are included
        assert "https://example.com/lesson1" in result
        assert "https://example.com/lesson2" in result

    def test_execute_missing_course_fields(self, mock_vector_store):
        """Test handling of missing course fields"""
        # Mock course with minimal data
        mock_vector_store.get_all_courses_metadata.return_value = [{
            "title": "Minimal Course",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Basic Lesson"}
            ]
        }]
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Advanced Retrieval")
        
        # Should handle missing instructor and course link gracefully
        assert "**Course:** Minimal Course" in result
        assert "**Instructor:**" not in result  # Should not include missing instructor
        assert "**Course Link:**" not in result  # Should not include missing link
        assert "Lesson 1: Basic Lesson" in result


class TestToolManager:
    """Test cases for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_register_multiple_tools(self, mock_vector_store):
        """Test registering multiple tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        assert len(manager.tools) == 2
        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        tool_names = [defn["name"] for defn in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Test tool execution via manager"""
        mock_vector_store.search.return_value = sample_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        # Should contain formatted search results
        assert "Advanced Retrieval for AI with Chroma" in result

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test getting sources from last search operation"""
        mock_vector_store.search.return_value = sample_search_results
        
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test query")
        
        sources = manager.get_last_sources()
        
        assert len(sources) == 2
        assert sources[0]["text"] == "Advanced Retrieval for AI with Chroma - Lesson 1"

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources from all tools"""
        mock_vector_store.search.return_value = sample_search_results
        
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test query")
        
        # Verify sources exist
        assert len(manager.get_last_sources()) == 2
        
        # Reset sources
        manager.reset_sources()
        
        # Verify sources are cleared
        assert len(manager.get_last_sources()) == 0

    def test_register_tool_without_name(self, mock_vector_store):
        """Test registering a tool without a name raises error"""
        manager = ToolManager()
        
        # Create a mock tool with no name in definition
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "test"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)