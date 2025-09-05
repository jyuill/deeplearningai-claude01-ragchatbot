"""
Integration tests for AI Generator tool calling and response generation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGenerator:
    """Test cases for AIGenerator"""

    def test_init(self, test_config):
        """Test AIGenerator initialization"""
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        assert generator.model == test_config.ANTHROPIC_MODEL
        assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class, test_config):
        """Test response generation without tool calling"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="This is a general knowledge answer")]
        mock_client.messages.create.return_value = mock_response
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        response = generator.generate_response("What is 2+2?")
        
        # Verify
        assert response == "This is a general knowledge answer"
        mock_client.messages.create.assert_called_once()
        
        # Check that no tools were provided
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" not in call_args

    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_use(self, mock_anthropic_class, test_config, mock_vector_store):
        """Test response generation with tools available but not used"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="This is a general knowledge answer")]
        mock_client.messages.create.return_value = mock_response
        
        # Setup tool manager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        response = generator.generate_response(
            "What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify
        assert response == "This is a general knowledge answer"
        mock_client.messages.create.assert_called_once()
        
        # Check that tools were provided
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert len(call_args["tools"]) > 0

    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class, test_config, mock_vector_store, sample_search_results):
        """Test response generation with tool calling"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock initial response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "What are vector databases?"}
        
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [mock_tool_block]
        
        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Vector databases are systems that store and query high-dimensional vectors.")]
        
        # Configure client to return initial response first, then final response
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Setup tool manager with mock vector store
        mock_vector_store.search.return_value = sample_search_results
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        response = generator.generate_response(
            "What are vector databases?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify
        assert response == "Vector databases are systems that store and query high-dimensional vectors."
        assert mock_client.messages.create.call_count == 2  # Initial + final calls
        
        # Verify tool was called
        mock_vector_store.search.assert_called_once_with(
            query="What are vector databases?",
            course_name=None,
            lesson_number=None
        )

    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class, test_config):
        """Test response generation with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Continuing our conversation...")]
        mock_client.messages.create.return_value = mock_response
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        response = generator.generate_response(
            "What about ChromaDB?",
            conversation_history="User: What are vector databases?\nAssistant: Vector databases store embeddings."
        )
        
        # Verify
        assert response == "Continuing our conversation..."
        
        # Check that system content includes conversation history
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]

    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_class, test_config, mock_vector_store, sample_search_results):
        """Test handling multiple tool calls in single response"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock initial response with multiple tool uses
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "vector databases"}
        
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"query": "embeddings"}
        
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [tool_block_1, tool_block_2]
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Combined response about vectors and embeddings.")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Setup tool manager
        mock_vector_store.search.return_value = sample_search_results
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        response = generator.generate_response(
            "Tell me about vectors and embeddings",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify both tool calls were made
        assert mock_vector_store.search.call_count == 2
        assert response == "Combined response about vectors and embeddings."

    @patch('anthropic.Anthropic')
    def test_tool_execution_with_tool_error(self, mock_anthropic_class, test_config, mock_vector_store, error_search_results):
        """Test tool execution when tool returns an error"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock initial response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test query"}
        
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [mock_tool_block]
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="I encountered an error searching for that information.")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Setup tool manager with error response
        mock_vector_store.search.return_value = error_search_results
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        response = generator.generate_response(
            "Find information about XYZ",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify error was handled gracefully
        assert response == "I encountered an error searching for that information."
        mock_vector_store.search.assert_called_once()

    def test_system_prompt_content(self, test_config):
        """Test that system prompt contains expected guidance"""
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        system_prompt = generator.SYSTEM_PROMPT
        
        # Check for key guidance elements
        assert "Course Outline Tool" in system_prompt
        assert "Content Search Tool" in system_prompt
        assert "One tool use per query maximum" in system_prompt
        assert "course structure" in system_prompt
        assert "lesson lists" in system_prompt

    @patch('anthropic.Anthropic')
    def test_api_parameters_format(self, mock_anthropic_class, test_config):
        """Test that API parameters are formatted correctly"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        generator.generate_response("Test query")
        
        # Verify API call parameters
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == test_config.ANTHROPIC_MODEL
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Test query"

    @patch('anthropic.Anthropic')
    def test_tool_choice_parameter(self, mock_anthropic_class, test_config, mock_vector_store):
        """Test that tool_choice parameter is set correctly when tools are provided"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        
        # Setup tools
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        generator.generate_response(
            "Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify tool_choice is set to auto
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["tool_choice"] == {"type": "auto"}

    @patch('anthropic.Anthropic')
    def test_non_tool_content_blocks_ignored(self, mock_anthropic_class, test_config, mock_vector_store):
        """Test that non-tool content blocks are ignored during tool execution"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response with mixed content (tool use + text)
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Here's what I found:"
        
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [text_block, tool_block]
        
        final_response = Mock()
        final_response.content = [Mock(text="Final response")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Setup tool manager
        mock_vector_store.search.return_value = Mock(error=None, is_empty=lambda: False)
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        response = generator.generate_response(
            "Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify only one tool call was made (text block ignored)
        mock_vector_store.search.assert_called_once()
        assert response == "Final response"