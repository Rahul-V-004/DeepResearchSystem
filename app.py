from flask import Flask, render_template, request, jsonify, url_for
import os
import json
import re
from datetime import datetime
from typing import Dict, Any

# Import the DeepResearchSystem from your existing code
from deep_research import DeepResearchSystem

app = Flask(__name__)

# Initialize the research system
research_system = DeepResearchSystem()

# Store research results in memory (for demo purposes)
# In a production app, you'd use a database
research_cache = {}

@app.route('/')
def index():
    """Render the main page with the search form"""
    return render_template('index.html')

@app.route('/research', methods=['POST'])
def perform_research():
    """Handle the research query submission"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty"}), 400
    
    # Generate a unique ID for this research
    research_id = f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # Process the query
        result = research_system.process_query(query)
        
        # Clean up the final answer before storing
        if "final_answer" in result:
            result["final_answer"] = clean_final_answer(result["final_answer"])
        
        # Store the result in our cache
        research_cache[research_id] = result
        
        return jsonify({
            "status": "success", 
            "research_id": research_id,
            "redirect": url_for('view_results', research_id=research_id)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def clean_final_answer(answer_text):
    """Clean up the final answer by removing links and formatting the text better"""
    # Remove URLs
    cleaned_text = answer_text #re.sub(r'https?://\S+', '', answer_text)
    
    # Remove source citations like [Source: URL]
    cleaned_text = re.sub(r'\[Source:.*?\]', '', cleaned_text)
    
    # Clean up any remaining brackets with citations
    cleaned_text = re.sub(r'\(Note:.*?\)', '', cleaned_text)
    
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    # Remove text between ** marks but keep the content
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)
    
    # Replace multiple newlines with a single newline
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    
    # Make sure paragraphs are properly formatted
    paragraphs = [p.strip() for p in cleaned_text.split('\n') if p.strip()]
    cleaned_text = '\n\n'.join(paragraphs)
    
    return cleaned_text

@app.route('/results/<research_id>')
def view_results(research_id):
    """Display the research results"""
    if research_id not in research_cache:
        return render_template('error.html', message="Research not found"), 404
    
    result = research_cache[research_id]
    return render_template('results.html', 
                          query=result["query"],
                          status=result["status"],
                          sub_queries=result.get("sub_queries", []),
                          final_answer=result.get("final_answer", ""),
                          sources=result.get("sources", []))

@app.route('/api/results/<research_id>')
def get_results_json(research_id):
    """API endpoint to get research results as JSON"""
    if research_id not in research_cache:
        return jsonify({"status": "error", "message": "Research not found"}), 404
    
    return jsonify(research_cache[research_id])

if __name__ == "__main__":
    app.run(debug=True)