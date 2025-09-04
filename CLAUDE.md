# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install Python dependencies using uv
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY
```

### Running the Application
```bash
# Quick start using the provided script
./run.sh

# Manual start (alternative)
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Testing and Development
The application serves both the API and frontend at:
- Web Interface: `http://localhost:8000`  
- API Documentation: `http://localhost:8000/docs`

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** with a full-stack architecture consisting of:

### Core Architecture Pattern
The system follows a **layered RAG architecture** where user queries flow through:
1. **Frontend** → **FastAPI** → **RAG System** → **AI Generator** → **Claude API**
2. **Conditional Search Branch**: AI Generator → Search Tools → Vector Store → ChromaDB

### Key Components and Their Relationships

**RAG System (`rag_system.py`)** - Central orchestrator that:
- Coordinates all components (document processor, vector store, AI generator, session manager)
- Manages the query processing pipeline
- Handles tool-based search integration
- Controls conversation history and source attribution

**AI Generator (`ai_generator.py`)** - Claude API integration that:
- Uses a static system prompt optimized for educational content
- Handles tool execution for search operations
- Manages conversation context and response synthesis
- Pre-builds API parameters for performance

**Search Tools (`search_tools.py`)** - Tool-based search system with:
- `CourseSearchTool` implementing the Tool abstract base class
- Anthropic tool definitions for Claude to use
- Smart course name matching and lesson filtering
- Source tracking for response attribution

**Vector Store (`vector_store.py`)** - ChromaDB integration featuring:
- Dual collection design: `course_catalog` (metadata) + `course_content` (chunks)
- Semantic search with sentence-transformers (`all-MiniLM-L6-v2`)
- Course title deduplication and smart filtering
- Structured search results with error handling

**Document Processor (`document_processor.py`)** - Text processing pipeline that:
- Extracts course metadata and lesson structure from transcript files
- Implements sentence-aware chunking with configurable overlap (800 chars, 100 overlap)
- Parses course titles, instructor names, and lesson hierarchies
- Creates `CourseChunk` objects with metadata for vector storage

**Session Manager (`session_manager.py`)** - Conversation state management:
- Creates unique session IDs for conversation tracking
- Maintains conversation history with configurable limits (max 2 exchanges)
- Provides formatted context for AI generation

### Data Models (`models.py`)
- `Course`: Full course metadata with lessons list
- `Lesson`: Individual lesson with number, title, and optional links  
- `CourseChunk`: Text chunk with course/lesson attribution for vector storage

### Configuration (`config.py`)
Centralized configuration using dataclasses and environment variables:
- Claude model: `claude-sonnet-4-20250514`
- Embedding model: `all-MiniLM-L6-v2` 
- Chunking: 800 chars with 100 char overlap
- Search: Max 5 results, 2 conversation exchanges

### Frontend Integration
- **Vanilla JavaScript** SPA with no frameworks
- **API Integration**: Two endpoints - `/api/query` (chat) and `/api/courses` (stats)
- **Session Management**: Handles session creation and conversation continuity
- **Real-time UI**: Loading states, markdown rendering, collapsible sources

## Key Implementation Details

### Vector Storage Strategy
The system uses a **dual-collection ChromaDB design**:
- `course_catalog`: Stores course-level metadata for semantic course discovery
- `course_content`: Stores text chunks with hierarchical metadata (course + lesson)

### Tool-Based Search Architecture  
Claude decides autonomously whether to search based on query analysis:
- **General knowledge**: Answered without search
- **Course-specific queries**: Triggers semantic search via tools
- **One search per query maximum** to optimize performance

### Document Processing Pipeline
Course transcripts are processed with:
1. **Metadata extraction**: Course title, instructor, lesson structure
2. **Content chunking**: Sentence-aware splitting with overlap
3. **Hierarchical organization**: Course → Lesson → Chunks with metadata preservation

### Performance Optimizations
- **Pre-built API parameters** in AI Generator to reduce request preparation overhead
- **Static system prompts** to avoid string concatenation on each call  
- **Course deduplication** to prevent re-processing existing content
- **Chunking strategy** balances context preservation with search granularity

## Development Notes

### Adding New Course Content
Place course documents (`.txt`, `.pdf`, `.docx`) in the `/docs` folder. The system will automatically process them on startup and avoid re-processing existing courses.

### Modifying Search Behavior
Search logic is centralized in `vector_store.py` with the `search()` method. Course name matching uses fuzzy string matching for flexible queries.

### Extending Tool Functionality
New tools should inherit from the `Tool` abstract base class in `search_tools.py` and be registered with the `ToolManager`.

### Configuration Changes
All configurable parameters are in `config.py`. Changes to chunk size, embedding models, or API settings require application restart.

### ChromaDB Persistence
The vector database persists to `./backend/chroma_db/` (gitignored). Delete this directory to force a complete rebuild of the vector store.
- Always use descriptive variable names
- always use uv to run the server do not use pip directly
- always use uv to manage all dependencies
- use uv to run python files