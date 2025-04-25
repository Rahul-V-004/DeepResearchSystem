from flask import Flask, render_template, request, jsonify, url_for
import os
import json
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
        
        # Store the result in our cache
        research_cache[research_id] = result
        
        return jsonify({
            "status": "success", 
            "research_id": research_id,
            "redirect": url_for('view_results', research_id=research_id)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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