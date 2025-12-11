"""LLM service using Ollama - Enhanced with multi-strategy classification"""
import ollama
import logging
import re
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)


class LLMService:
    """Handles all LLM operations with optimized classification"""
    
    # Keywords for intelligent multi-strategy classification
    CATEGORY_KEYWORDS = {
        "Code": {
            "strong": ["def ", "class ", "function", "import ", "return", "algorithm", "implementation",
                      "variable", "loop", "array", "object", "method", "module", "package", "library",
                      "syntax", "compile", "debug", "exception", "try", "except", "interface", "type hint"],
            "weak": ["programming", "developer", "code", "software", "technical"]
        },
        "Documentation": {
            "strong": ["## ", "# ", "api", "endpoint", "rest", "http", "json", "schema", "response",
                      "parameter", "authentication", "authorization", "oauth", "guide", "tutorial",
                      "usage", "example", "reference", "specification", "documentation"],
            "weak": ["help", "explain", "describe", "design", "architecture", "pattern"]
        },
        "Education": {
            "strong": ["question", "answer", "quiz", "exercise", "test", "exam", "learning", "course",
                      "lesson", "assignment", "homework", "solution", "evaluate", "understand",
                      "concept", "theory", "principle"],
            "weak": ["teaching", "study", "student", "educational", "learn"]
        },
        "Technology": {
            "strong": ["ai", "ml", "machine learning", "neural", "algorithm", "data", "cloud", "deployment",
                      "kubernetes", "docker", "devops", "infrastructure", "framework", "tool", "platform",
                      "innovation", "technology", "system", "performance", "optimization"],
            "weak": ["tech", "tools", "systems", "software"]
        },
        "Business": {
            "strong": ["business", "strategy", "marketing", "sales", "revenue", "profit", "customer",
                      "market", "growth", "operations", "management", "leadership", "roi", "financial",
                      "investment", "enterprise", "organization"],
            "weak": ["company", "work", "plan", "goal", "objective"]
        },
        "Other": {
            "strong": [],
            "weak": []
        }
    }
    
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        logger.info(f"LLM Service initialized with model: {model}")
    
    def _analyze_keywords(self, text: str) -> Dict[str, int]:
        """Analyze keyword distribution across categories"""
        text_lower = text.lower()
        scores = {category: 0 for category in self.CATEGORY_KEYWORDS}
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            # Strong keyword matches (2 points each)
            for keyword in keywords["strong"]:
                scores[category] += text_lower.count(keyword) * 2
            
            # Weak keyword matches (1 point each)
            for keyword in keywords["weak"]:
                scores[category] += text_lower.count(keyword) * 1
        
        return scores
    
    def _analyze_structure(self, text: str) -> Dict[str, float]:
        """Analyze document structure to infer category"""
        lines = text.split('\n')
        scores = {category: 0.0 for category in self.CATEGORY_KEYWORDS}
        
        # Header analysis (more headers = documentation)
        header_count = sum(1 for line in lines if line.strip().startswith('#'))
        if header_count > 5:
            scores["Documentation"] += 3
        
        # Code pattern analysis - check for programming constructs
        code_indicators = ['def ', 'class ', 'import ', 'function', 'return', 'if ', 'for ', 'while ']
        code_lines = sum(1 for line in lines if any(ind in line for ind in code_indicators))
        if code_lines > len(lines) * 0.2:  # 20%+ lines have code patterns
            scores["Code"] += 4
        
        # Question/Answer analysis
        qa_patterns = ['?', 'question:', 'answer:', 'q:', 'a:', 'what ', 'how ', 'why ']
        qa_count = sum(1 for line in lines if any(pat in line.lower() for pat in qa_patterns))
        if qa_count > len(lines) * 0.15:
            scores["Education"] += 3
        
        # JSON/Schema analysis (API/Documentation)
        if re.search(r'\{.*\}', text) and re.search(r'"[^"]*":\s*', text):
            scores["Documentation"] += 2
        
        # Code block indicators
        if '```' in text or '```python' in text or '```javascript' in text:
            scores["Code"] += 3
        
        # Structured list patterns (documentation)
        if text.count('\n-') > 10:
            scores["Documentation"] += 2
        
        return scores
    
    def _analyze_content_type(self, text: str) -> Dict[str, float]:
        """Analyze specific content patterns and semantic meaning"""
        scores = {category: 0.0 for category in self.CATEGORY_KEYWORDS}
        text_lower = text.lower()
        
        # API endpoint patterns (REST)
        if re.search(r'(get|post|put|delete|patch)\s+(/[a-z/]+|"[a-z/]+")', text_lower):
            scores["Documentation"] += 3
        
        # Algorithm/complexity analysis
        if re.search(r'(time complexity|space complexity|o\(|algorithm|computational)', text_lower):
            scores["Code"] += 2
        
        # Business metrics and KPIs
        if re.search(r'(roi|kpi|revenue|profit|market|customer|stakeholder|budget)', text_lower):
            scores["Business"] += 2
        
        # ML/AI patterns (neural networks, training, etc)
        if re.search(r'(neural|network|training|model|dataset|tensor|epoch|layer|gradient)', text_lower):
            scores["Technology"] += 3
        
        # Learning and assessment patterns
        if re.search(r'(chapter|section|quiz|exercise|solution|evaluate|understand|homework)', text_lower):
            scores["Education"] += 2
        
        # SQL/Database patterns
        if re.search(r'(select|insert|update|delete|table|database|schema|query)', text_lower):
            scores["Code"] += 2
        
        # Configuration file patterns
        if re.search(r'(config|yaml|json|\.env|environment|setting)', text_lower):
            scores["Technology"] += 1
        
        return scores
    
    def _classify_by_analysis(self, text: str) -> Tuple[str, float]:
        """Use multi-strategy analysis to classify content - fast and accurate"""
        keyword_scores = self._analyze_keywords(text)
        structure_scores = self._analyze_structure(text)
        content_scores = self._analyze_content_type(text)
        
        # Combine all scores with strategic weighting
        final_scores = {}
        for category in self.CATEGORY_KEYWORDS:
            final_scores[category] = (
                keyword_scores.get(category, 0) * 1.0 +      # Keywords (1x weight)
                structure_scores.get(category, 0) * 1.5 +    # Structure (1.5x weight)
                content_scores.get(category, 0) * 1.5        # Content patterns (1.5x weight)
            )
        
        # Get best category
        best_category = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_category]
        
        # Filter out "Other" if we have real candidates
        if best_category == "Other" and best_score == 0:
            return "Other", 0.0
        
        logger.debug(f"Classification scores: {final_scores}")
        return best_category, best_score
    
    def classify_content(self, text: str) -> str:
        """Classify content using optimized multi-strategy approach
        
        Strategy:
        1. Fast analysis using keywords, structure, and content patterns
        2. If high confidence (score > 15), return immediately
        3. If low confidence, use LLM for verification
        4. Fallback to analysis if LLM unclear
        
        This optimizes for speed (most docs > 15 score) while maintaining accuracy.
        """
        try:
            # Step 1: Fast content analysis (no LLM needed)
            analysis_category, analysis_score = self._classify_by_analysis(text)
            
            # Step 2: High confidence threshold - use analysis result
            if analysis_score > 15:
                logger.info(f"✓ FAST CLASSIFIED (score: {analysis_score:.1f}): {analysis_category}")
                return analysis_category
            
            # Step 3: Low confidence - verify with LLM
            logger.info(f"⚠ Low confidence ({analysis_score:.1f}), using LLM verification...")
            
            prompt = f"""You are a document classifier. Classify into ONE category:

- Code: software code, algorithms, implementations, functions, classes
- Documentation: API docs, guides, tutorials, references, specifications  
- Education: questions, exercises, tests, courses, learning materials
- Technology: tech news, tools, infrastructure, cloud, AI/ML, DevOps
- Business: strategy, operations, sales, marketing, management, ROI
- Other: doesn't fit above

Respond with ONLY the category name.

Document:
{text[:800]}

Category:"""
            
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.05,
                    "num_predict": 5
                }
            )
            
            raw_response = response['response'].strip().lower()
            first_word = raw_response.split()[0] if raw_response else "other"
            
            category_map = {
                "code": "Code",
                "documentation": "Documentation",
                "docs": "Documentation",
                "api": "Documentation",
                "education": "Education",
                "educational": "Education",
                "learning": "Education",
                "questions": "Education",
                "technology": "Technology",
                "tech": "Technology",
                "tools": "Technology",
                "business": "Business",
                "operations": "Business",
                "other": "Other"
            }
            
            for key, value in category_map.items():
                if key in first_word:
                    logger.info(f"✓ LLM VERIFIED: {value}")
                    return value
            
            # Step 4: Fallback to analysis if LLM unclear
            logger.info(f"✓ LLM unclear, using analysis: {analysis_category}")
            return analysis_category
            
        except Exception as e:
            logger.error(f"Error classifying content: {e}")
            try:
                analysis_category, _ = self._classify_by_analysis(text)
                return analysis_category
            except:
                return "Other"
    
    def _calculate_confidence(self, query: str, chunks: List[dict]) -> float:
        """Calculate confidence score (0-100) for the answer"""
        if not chunks:
            return 0.0
        
        avg_similarity = sum(chunk.get('similarity', 0) for chunk in chunks) / len(chunks)
        chunk_bonus = min(len(chunks) / 5.0, 1.0)
        avg_distance = sum(chunk.get('distance', 2.0) for chunk in chunks) / len(chunks)
        distance_confidence = max(0, 1.0 - (avg_distance / 2.0))
        
        confidence = (avg_similarity * 0.4 + chunk_bonus * 0.3 + distance_confidence * 0.3)
        confidence_score = int(confidence * 100)
        return max(0, min(100, confidence_score))
    
    def _get_confidence_level(self, score: int) -> str:
        """Get confidence level label"""
        if score >= 80:
            return "🟢 HIGH"
        elif score >= 50:
            return "🟡 MEDIUM"
        else:
            return "🔴 LOW"
    
    def generate_response(self, query: str, context_chunks: List[dict]) -> Tuple[str, List[str], float, List[dict]]:
        """Generate response with confidence score and source snippets"""
        
        if not context_chunks:
            return "I cannot find this information in your local documents.", [], 0, []
        
        context_chunks = context_chunks[:5]
        confidence_score = self._calculate_confidence(query, context_chunks)
        confidence_level = self._get_confidence_level(confidence_score)
        
        source_snippets = []
        for i, chunk in enumerate(context_chunks, 1):
            snippet = {
                'id': i,
                'filename': chunk['filename'],
                'category': chunk.get('category', 'Unknown'),
                'text': chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text'],
                'similarity': chunk.get('similarity', 0),
                'relevance_pct': int(chunk.get('similarity', 0) * 100)
            }
            source_snippets.append(snippet)
        
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = f"[Source {i}: {chunk['filename']}]"
            context_parts.append(f"{source_info}\n{chunk['text']}\n")
        
        context_text = "\n".join(context_parts)
        
        full_prompt = f"""You are a helpful AI assistant that answers questions STRICTLY based on the provided documents.

IMPORTANT RULES:
- ONLY use information from the provided documents below
- If the answer is NOT in the documents, respond: "I don't have that information in the provided documents. Please ask a question related to the available documents."
- Do NOT make up information or use external knowledge
- Cite which document each piece of information comes from

Documents:
{context_text}

Question: {query}

Answer based ONLY on the documents above:"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                stream=False,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 350,
                    "num_ctx": 3072,
                    "repeat_penalty": 1.1,
                    "num_thread": 8,
                }
            )
            
            answer = response['response'].strip()
            
            # Check if LLM says information is not in documents
            no_info_phrases = [
                "don't have that information",
                "not in the provided documents",
                "cannot find this information",
                "not found in the documents",
                "no information about",
                "not mentioned in the documents"
            ]
            
            is_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
            
            if is_no_info:
                # Return clean response without sources or confidence
                return answer, [], 0, []
            
            cited_files = list(set([chunk['filename'] for chunk in context_chunks]))
            
            if cited_files:
                answer += f"\n\n📊 Confidence: {confidence_level} ({confidence_score}%)"
                answer += f"\n📄 Sources: {', '.join(cited_files)}"
            
            return answer, cited_files, confidence_score, source_snippets
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", [], 0, []
    
    def check_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            ollama.list()
            return True
        except:
            return False
