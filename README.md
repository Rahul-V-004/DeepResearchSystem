# Deep Research System - Flask Web Application

A Flask web application that provides an intuitive interface for the DeepResearchSystem, allowing users to submit complex research queries and receive comprehensive, well-cited answers based on real-time web searches.

## Features

- **Advanced Research Processing**: Decomposes complex queries into sub-queries for comprehensive research
- **Web-based Interface**: Clean, responsive UI built with Bootstrap
- **Real-time Web Search**: Leverages Tavily Search API for current information
- **AI-powered Synthesis**: Uses Google Gemini 1.5 Pro to analyze and synthesize research findings
- **Citation Support**: Includes citations for all information in the final answer
- **Vector Storage**: Saves search results for efficient retrieval and context management
- **API Access**: JSON API endpoint for programmatic access to research results

## Prerequisites

- Python 3.8+
- Google API key (for Gemini 1.5 Pro)
- Tavily API key (for web search)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deep-research-flask.git
   cd deep-research-flask
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your-google-api-key
   TAVILY_API_KEY=your-tavily-api-key
   ```

## Project Structure

```
deep_research_flask/
├── app.py                  # The main Flask application file
├── deep_research.py        # DeepResearchSystem implementation
├── requirements.txt        # Dependencies
├── .env                    # Environment variables (API keys)
└── templates/              # Flask HTML templates
    ├── index.html          # Search form page
    ├── results.html        # Research results page
    └── error.html          # Error page
```

## Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Enter your research query in the form and submit.

## How It Works

1. **Query Submission**: User submits a research query through the web interface
2. **Query Decomposition**: The system breaks down the query into 3-5 specific sub-queries
3. **Information Gathering**: For each sub-query, the system performs a web search using Tavily API
4. **Vector Storage**: Search results are stored in Chroma vector database with embeddings
5. **Answer Synthesis**: Google Gemini 1.5 Pro generates a comprehensive answer based on all gathered information
6. **Final Polish**: The draft answer is reviewed and improved for clarity and proper citations
7. **Result Display**: The final answer and sources are presented to the user in a clean, organized format

## API Usage

Access research results programmatically:

```
GET /api/results/<research_id>
```

Example response:
```json
{
  "status": "success",
  "query": "What are the latest developments in quantum computing?",
  "sub_queries": ["Recent breakthroughs in quantum computing hardware", "Quantum computing software developments", "Practical applications of quantum computing"],
  "final_answer": "Quantum computing has seen significant advancements in recent years...",
  "sources": [...]
}
```

## Customization

- **Model Selection**: Change the model name in `ResearchAgent` and `AnswerAgent` initialization
- **UI Customization**: Modify the templates in the `templates` directory
- **Vector Storage**: Configure Chroma settings in the `ResearchAgent` class

## Limitations

- In-memory storage of research results (not persistent)
- No user authentication or result history
- Processing time depends on API response times
- Limited error handling for API failures

## Future Improvements

- Database integration for persistent storage
- User accounts and saved research history
- Progress updates during research processing
- Enhanced error handling and retry mechanisms
- Export functionality for research results (PDF, DOCX)
- Custom citation style options

## License

[MIT License](LICENSE)

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the agent framework
- [Tavily](https://tavily.com/) for the search API
- [Google Gemini](https://ai.google.dev/) for the AI model
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the UI components
