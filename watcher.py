"""
Universal RAG System - File Watcher & Processing Backend (Refactored)
Clean, modular architecture with separation of concerns
"""

import os
import time
import shutil
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from core import DatabaseManager, LLMService, FileProcessor
from models import Document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INCOMING_DIR = DATA_DIR / "incoming"
SORTED_DIR = DATA_DIR / "sorted"
DB_DIR = DATA_DIR / "database"

# Ensure directories exist
INCOMING_DIR.mkdir(parents=True, exist_ok=True)
SORTED_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# Initialize services
db_manager = DatabaseManager(DB_DIR)
llm_service = LLMService(model='llama3.2')
file_processor = FileProcessor()

# Track processed files
processed_files = {}

def process_file(filepath):
    """Process a single file: extract, chunk, classify, store"""
    filepath = Path(filepath)
    
    if not filepath.exists() or not filepath.is_file():
        return
    
    logger.info(f"Processing file: {filepath.name}")
    
    try:
        # Extract text
        text = file_processor.extract_text(filepath)
        if not text:
            text = f"File: {filepath.name}"
        
        logger.info(f"Extracted {len(text)} characters from {filepath.name}")
        
        # Classify content
        category = llm_service.classify_content(text)
        logger.info(f"Classified as: {category}")
        
        # Create document object
        document = file_processor.create_document(filepath, text, category)
        
        # Create category directory
        category_dir = SORTED_DIR / category
        category_dir.mkdir(exist_ok=True)
        
        # Move file to sorted directory
        dest_path = category_dir / filepath.name
        
        # Handle duplicate filenames
        counter = 1
        while dest_path.exists():
            dest_path = category_dir / f"{filepath.stem}_{counter}{filepath.suffix}"
            counter += 1
        
        shutil.move(str(filepath), str(dest_path))
        logger.info(f"Moved to: {dest_path}")
        
        # Update document filepath
        document.filepath = dest_path
        
        # Create chunks with better context preservation
        chunks = file_processor.create_chunks(document, chunk_size=600)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Store in ChromaDB
        if chunks:
            db_manager.add_chunks(chunks)
            
            # Track processed file
            processed_files[document.file_hash] = {
                'filename': document.filename,
                'path': str(dest_path),
                'chunk_ids': [chunk.chunk_id for chunk in chunks]
            }
            
            logger.info(f"Added {len(chunks)} chunks to database")
        
        logger.info(f"✓ Successfully processed: {filepath.name}")
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")

def remove_file_from_db(filepath):
    """Remove file vectors from database when file is deleted"""
    filepath = Path(filepath)
    
    try:
        # Find file hash in processed files
        file_hash = None
        for hash_key, info in processed_files.items():
            if info['path'] == str(filepath):
                file_hash = hash_key
                break
        
        if file_hash:
            deleted_count = db_manager.delete_by_hash(file_hash)
            
            if file_hash in processed_files:
                del processed_files[file_hash]
            
            logger.info(f"Removed {deleted_count} chunks from database for {filepath.name}")
        
    except Exception as e:
        logger.error(f"Error removing file from database: {e}")



class FileWatcherHandler(FileSystemEventHandler):
    """Handle file system events"""
    
    def on_created(self, event):
        if not event.is_directory:
            # Wait a bit to ensure file is fully written
            time.sleep(1)
            logger.info(f"New file detected: {event.src_path}")
            process_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            remove_file_from_db(event.src_path)


def process_existing_files():
    """Process any existing files in incoming directory"""
    logger.info("Checking for existing files in incoming directory...")
    
    for filepath in INCOMING_DIR.iterdir():
        if filepath.is_file():
            process_file(filepath)


def start_watching():
    """Start the file watcher"""
    logger.info("=" * 60)
    logger.info("DocuMind AI - File Watcher Started")
    logger.info("=" * 60)
    logger.info(f"Monitoring: {INCOMING_DIR}")
    logger.info(f"Database: {DB_DIR}")
    logger.info(f"Sorted files: {SORTED_DIR}")
    logger.info("=" * 60)
    
    # Process existing files first
    process_existing_files()
    
    # Setup watchdog
    event_handler = FileWatcherHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INCOMING_DIR), recursive=False)
    observer.start()
    
    logger.info("✓ File watcher is active. Drop files into data/incoming/")
    logger.info("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()
    
    observer.join()
    logger.info("File watcher stopped.")


if __name__ == "__main__":
    start_watching()
