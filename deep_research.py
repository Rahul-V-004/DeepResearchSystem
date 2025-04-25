import os
from typing import List, Dict, Any, Optional, TypedDict
from enum import Enum
from datetime import datetime

import langchain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END


# With these
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from the environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Still needed for search
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   # For Gemini API

# Remove OPENAI_API_KEY as it's no longer needed

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)



DEFAULT_MODEL = "gemini-1.5-pro"

# Define the State using TypedDict for compatibility with LangGraph
class AgentState(TypedDict, total=False):
    """State for the research workflow."""
    query: str
    sub_queries: List[str]
    research_results: List[Dict[str, Any]]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    error_msg: Optional[str]

# Research Agent - Handles query decomposition and information gathering
class ResearchAgent:
    def __init__(self, model_name="gemini-1.5-pro"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
        # Use Google's embeddings instead of OpenAI
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = Chroma(embedding_function=self.embeddings, collection_name="research_results")
        
    # Rest of the class remains the same
        
    def decompose_query(self, query: str) -> List[str]:
        """Break down a complex query into smaller, searchable sub-queries."""
        # Create messages directly instead of using a template
        messages = [
            SystemMessage(content="You are an expert research planner. Your task is to decompose a complex research query into 3-5 specific sub-queries that will help gather comprehensive information on the topic. Focus on different aspects of the main query."),
            HumanMessage(content=f"Decompose the following research query into 3-5 specific sub-queries: {query}")
        ]
        
        try:
            # Invoke the LLM directly with messages
            response = self.llm.invoke(messages)
            
            # Extract potential list from the text
            lines = response.content.strip().split('\n')
            sub_queries = [line.strip('- ').strip() for line in lines if line.strip().startswith('-')]
            
            if not sub_queries:
                # Try additional extraction for numbered lists
                sub_queries = [line[line.find('.') + 1:].strip() for line in lines 
                               if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10))]
            
            if not sub_queries:
                # Fallback to a simpler approach
                return [
                    f"Key aspects of {query}",
                    f"Recent developments in {query}",
                    f"Challenges and limitations of {query}"
                ]
            return sub_queries
        except Exception as e:
            print(f"Query decomposition error: {str(e)}")
            # Fallback to default sub-queries
            return [
                f"Key aspects of {query}",
                f"Recent developments in {query}",
                f"Challenges and limitations of {query}"
            ]
    
    def search_and_collect(self, sub_query: str) -> Dict[str, Any]:
        """Perform a search for a sub-query and collect the results."""
        try:
            search_results = self.search_tool.invoke({"query": sub_query})
            
            # Store results in vector database for retrieval
            docs = [
                Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "title": result.get("title", ""),
                        "source": result.get("url", ""),
                        "sub_query": sub_query,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                for result in search_results
            ]
            
            if docs:
                self.vector_store.add_documents(docs)
            
            return {
                "sub_query": sub_query,
                "results": search_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "sub_query": sub_query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process(self, state: AgentState) -> AgentState:
        """Process the state and perform research operations."""
        try:
            # Make a copy of the current state to modify
            new_state = state.copy()
            
            # If sub-queries not yet generated, decompose the query
            if "sub_queries" not in new_state or not new_state["sub_queries"]:
                new_state["sub_queries"] = self.decompose_query(new_state["query"])
            
            # Initialize research_results if it doesn't exist
            if "research_results" not in new_state:
                new_state["research_results"] = []
                
            # Perform search for each sub-query
            for sub_query in new_state["sub_queries"]:
                # Check if we've already processed this sub-query
                if not any(r.get("sub_query") == sub_query for r in new_state["research_results"]):
                    result = self.search_and_collect(sub_query)
                    new_state["research_results"].append(result)
            
            return new_state
        except Exception as e:
            return {**state, "error_msg": f"Research agent error: {str(e)}"}

# Answer Agent - Synthesizes information and drafts answers
class AnswerAgent:
    def __init__(self, model_name="gemini-1.5-pro"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        # Use Google's embeddings instead of OpenAI
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = Chroma(embedding_function=self.embeddings, collection_name="research_results")
    
    # Rest of the class remains the same
    
    def retrieve_relevant_info(self, query: str, k: int = 10) -> List[Document]:
        """Retrieve relevant information from the vector store."""
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception:
            # Return empty list if vector store is empty or error occurs
            return []
    
    def draft_answer(self, query: str, research_results: List[Dict[str, Any]]) -> str:
        """Draft an answer based on the research results."""
        # Get additional context from vector store
        relevant_docs = self.retrieve_relevant_info(query)
        
        # Prepare context from research results
        context = []
        for result in research_results:
            sub_query = result.get("sub_query", "")
            for item in result.get("results", []):
                context.append(f"Source: {item.get('url', 'Unknown')}\nTitle: {item.get('title', 'No title')}\nRelated to sub-query: {sub_query}\nContent: {item.get('content', 'No content')}")
        
        # Add vector store results
        for doc in relevant_docs:
            context.append(f"Source: {doc.metadata.get('source', 'Unknown')}\nTitle: {doc.metadata.get('title', 'No title')}\nContent: {doc.page_content}")
        
        # Create messages directly
        context_str = "\n\n".join(context[:15])  # Limit context to prevent token overflow
        messages = [
            SystemMessage(content="You are an expert research synthesizer. Your task is to create a comprehensive, accurate answer to the query based on the provided research results. Include citations in [Source: URL] format after each piece of information. Organize the answer logically and make it easy to understand."),
            HumanMessage(content=f"Query: {query}\n\nResearch Results:\n{context_str}")
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def process(self, state: AgentState) -> AgentState:
        """Process the state and draft an answer."""
        try:
            # Make a copy of the current state to modify
            new_state = state.copy()
            
            if "research_results" in new_state and new_state["research_results"] and \
               ("draft_answer" not in new_state or not new_state["draft_answer"]):
                new_state["draft_answer"] = self.draft_answer(new_state["query"], new_state["research_results"])
                
                # Create messages for final polish
                messages = [
                    SystemMessage(content="You are an editor specializing in academic and research content. Review the draft answer, ensure all statements are properly cited, and improve the overall structure and clarity. Maintain all citations in [Source: URL] format."),
                    HumanMessage(content=f"Original query: {new_state['query']}\n\nDraft answer to review and improve:\n{new_state['draft_answer']}")
                ]
                
                response = self.llm.invoke(messages)
                new_state["final_answer"] = response.content
            
            return new_state
        except Exception as e:
            return {**state, "error_msg": f"Answer agent error: {str(e)}"}

# Orchestration with LangGraph
def create_research_graph():
    """Create the research workflow graph."""
    # Initialize agents
    research_agent = ResearchAgent()
    answer_agent = AnswerAgent()
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Define conditional edge based on error state
    def should_continue_to_answer(state):
        return "error_msg" not in state or state["error_msg"] is None
    
    # Add nodes
    workflow.add_node("research", research_agent.process)
    workflow.add_node("answer", answer_agent.process)
    
    # Define edges with conditions
    workflow.add_conditional_edges(
        "research",
        should_continue_to_answer,
        {
            True: "answer",
            False: END
        }
    )
    workflow.add_edge("answer", END)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    return workflow.compile()

# Main system class
class DeepResearchSystem:
    def __init__(self):
        self.graph = create_research_graph()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a research query and return the results."""
        initial_state: AgentState = {"query": query}
        
        try:
            result = self.graph.invoke(initial_state)
            
            if "error_msg" in result and result["error_msg"]:
                return {
                    "status": "error",
                    "message": result["error_msg"],
                    "query": result["query"]
                }
            
            return {
                "status": "success",
                "query": result["query"],
                "sub_queries": result.get("sub_queries", []),
                "final_answer": result.get("final_answer", "No answer could be generated."),
                "sources": [r.get("results", []) for r in result.get("research_results", []) 
                           if "error" not in r]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"System error: {str(e)}",
                "query": query
            }

# Example usage
if __name__ == "__main__":
    # Set up environment
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Check for API keys
    # if TAVILY_API_KEY == "your-tavily-api-key":
    #     print("Warning: Please set your TAVILY_API_KEY environment variable")
    # if OPENAI_API_KEY == "your-openai-api-key":
    #     print("Warning: Please set your OPENAI_API_KEY environment variable")
    
    try:
        system = DeepResearchSystem()
        print("Processing query...")
        result = system.process_query("What are the latest developments in renewable energy technologies?")
        
        if result["status"] == "success":
            print("\n=== FINAL ANSWER ===\n")
            print(result["final_answer"])
            print("\n=== SOURCES ===\n")
            print(f"Found {len(result['sources'])} source collections")
        else:
            print(f"Error: {result['message']}")
    except Exception as e:
        print(f"System initialization error: {str(e)}")
        print("Check your API keys and dependencies.")