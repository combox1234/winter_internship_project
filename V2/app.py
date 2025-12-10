"""
Universal RAG System - Flask Application (Refactored)
Clean, modular architecture
"""

from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import logging
import os

from core import DatabaseManager, LLMService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "database"
SORTED_DIR = DATA_DIR / "sorted"

# Initialize services
db_manager = DatabaseManager(DB_DIR)
llm_service = LLMService(model='llama3.2')

logger.info(f"Database initialized with {db_manager.get_count()} documents")


@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Empty query'}), 400
        
        # Retrieve more chunks for higher accuracy (optimized chunking makes this efficient)
        chunks = db_manager.query(query, n_results=7)  # Optimal balance of accuracy and speed
        
        # Generate response
        answer, cited_files = llm_service.generate_response(query, chunks)
        
        return jsonify({
            'answer': answer,
            'cited_files': cited_files,
            'chunks_retrieved': len(chunks)
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<path:filename>')
def download_file(filename):
    """Download cited file"""
    try:
        # Search for file in sorted directory
        for root, dirs, files in os.walk(SORTED_DIR):
            if filename in files:
                filepath = os.path.join(root, filename)
                return send_file(filepath, as_attachment=True)
        
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """Get system status"""
    try:
        doc_count = db_manager.get_count()
        
        # Count files in sorted directory
        sorted_files = 0
        categories = []
        for category_dir in SORTED_DIR.iterdir():
            if category_dir.is_dir():
                categories.append(category_dir.name)
                sorted_files += len(list(category_dir.glob('*')))
        
        return jsonify({
            'database_count': doc_count,
            'sorted_files': sorted_files,
            'categories': categories,
            'ollama_available': llm_service.check_availability()
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500


def check_ollama():
    """Check if Ollama is available"""
    try:
        ollama.list()
        return True
    except:
        return False


if __name__ == '__main__':
    logger.info("Starting DocuMind AI...")
    logger.info(f"Database directory: {DB_DIR}")
    logger.info(f"Documents in database: {db_manager.get_count()}")
    
    # Optimized settings for faster startup and performance
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        use_reloader=False,  # Disable auto-reload for faster startup
        threaded=True  # Enable threading for better performance
    )
