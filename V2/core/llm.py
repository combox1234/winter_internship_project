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
    
    def generate_response(self, query: str, context_chunks: List[dict]) -> Tuple[str, List[str]]:
        """Generate response using improved RAG with flexibility for production"""
        
        if not context_chunks:
            return "I cannot find this information in your local documents.", []
        
        # Build context from chunks with better formatting
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"Source {i} [{chunk['filename']}]:\n{chunk['text']}\n\n"
        
        # Optimized prompt for accuracy and speed
        full_prompt = f"""You are a precise AI assistant. Answer questions using ONLY the provided context.

RULES:
1. Use ONLY information from the context below
2. If the answer is not in context, say "I cannot find this information in the documents"
3. Be direct and concise
4. Structure your answer with bullet points or paragraphs for clarity
5. Do not add information beyond what's in the context

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER (be precise and cite specific information):"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                stream=False,
                options={
                    "temperature": 0.1,  # Lower = more focused and accurate
                    "top_p": 0.95,  # Higher quality token selection
                    "top_k": 40,  # Limit to best tokens for speed
                    "num_predict": 400,  # Allow longer, more complete answers
                    "num_ctx": 4096  # Larger context window for accuracy
                }
            )
            
            answer = response['response'].strip()
            
            # Check if answer actually found information
            no_info_phrases = ["cannot find", "not in", "no information", "not available", "don't have", "not found"]
            has_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
            
            if has_no_info:
                cited_files = []
            else:
                cited_files = list(set([chunk['filename'] for chunk in context_chunks]))
            
            return answer, cited_files
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response. Make sure Ollama is running with {self.model} model.", []
    
    def check_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            ollama.list()
            return True
        except:
            return False
