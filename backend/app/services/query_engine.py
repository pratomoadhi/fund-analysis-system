"""
Query engine service for RAG-based question answering
"""
from typing import Dict, Any, List, Optional
import time
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings
from app.services.vector_store import VectorStore
from app.services.metrics_calculator import MetricsCalculator
from sqlalchemy.orm import Session


class QueryEngine:
    """RAG-based query engine for fund analysis"""
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.metrics_calculator = MetricsCalculator(db)
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM"""
        if settings.GOOGLE_API_KEY:
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_GENERATION_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.0
            )
        else:
            # Fallback to local LLM
            return Ollama(model="llama2")
    
    async def process_query(
        self, 
        query: str, 
        fund_id: Optional[int] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using RAG
        
        Args:
            query: User question
            fund_id: Optional fund ID for context
            conversation_history: Previous conversation messages
            
        Returns:
            Response with answer, sources, and metrics
        """
        start_time = time.time()
        
        # Mockup fund id
        fund_id = fund_id or 1
        
        # Step 1: Classify query intent
        intent = await self._classify_intent(query)
        
        # Step 2: Retrieve relevant context from vector store
        filter_metadata = {"fund_id": fund_id} if fund_id else None
        relevant_docs = await self.vector_store.similarity_search(
            query=query,
            k=settings.TOP_K_RESULTS,
            filter_metadata=filter_metadata
        )
        
        # Step 3: Calculate metrics if needed
        metrics = None
        if intent == "calculation" and fund_id:
            metrics = self.metrics_calculator.calculate_all_metrics(fund_id)
        
        # Step 4: Generate response using LLM
        answer = await self._generate_response(
            query=query,
            context=relevant_docs,
            metrics=metrics,
            conversation_history=conversation_history or []
        )
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc["content"],
                    "metadata": {
                        k: v for k, v in doc.items() 
                        if k not in ["content", "score"]
                    },
                    "score": doc.get("score")
                }
                for doc in relevant_docs
            ],
            "metrics": metrics,
            "processing_time": round(processing_time, 2)
        }
    
    async def _classify_intent(self, query: str) -> str:
        """
        Classify query intent
        
        Returns:
            'calculation', 'definition', 'retrieval', or 'general'
        """
        query_lower = query.lower()
        
        # Definition keywords
        def_keywords = [
            "what does", "mean", "define", "explain", "definition", 
            "what is a", "what are"
        ]
        if any(keyword in query_lower for keyword in def_keywords):
            return "definition"
        
        # Calculation keywords
        calc_keywords = [
            "calculate", "what is the", "current", "dpi", "irr", "tvpi", 
            "rvpi", "pic", "paid-in capital", "return", "performance"
        ]
        if any(keyword in query_lower for keyword in calc_keywords):
            return "calculation"
        
        # Retrieval keywords
        ret_keywords = [
            "show me", "list", "all", "find", "search", "when", 
            "how many", "which"
        ]
        if any(keyword in query_lower for keyword in ret_keywords):
            return "retrieval"
        
        return "general"
    
    async def _generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Generate response using LLM"""
        
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1}]\n{doc['content']}"
            for i, doc in enumerate(context[:3])  # Use top 3 sources
        ])
        
        # Build metrics string
        metrics_str = ""
        if metrics:
            metrics_str = "\n\nAvailable Metrics:\n"
            for key, value in metrics.items():
                if value is not None:
                    # Format value to 2 decimal places if it's a number
                    formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                    metrics_str += f"- {key.upper()}: {formatted_value}\n"
        
        # Build conversation history string
        history_str = ""
        if conversation_history:
            history_str = "\n\nPrevious Conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                # Ensure role names are capitalized for clarity in the prompt text
                role = "User" if msg.get('role', '').lower() == 'user' else "Assistant"
                history_str += f"{role}: {msg.get('content', '')}\n"

        # --- FIX: Combine System Instruction into the User Prompt ---
        system_instruction = """You are a financial analyst assistant specializing in private equity fund performance.

Your role:
- Answer questions about fund performance using provided context, metrics, and history.
- Always use the provided metrics data when relevant.
- Explain complex financial terms in simple language.
- When citing document data, reference the source (e.g., [Source 1]).
- If the answer cannot be found in the context or metrics, state that clearly and politely.

Format your responses:
- Be concise but thorough.
- Use bullet points for lists.
- Bold important numbers and financial terms."""
        
        # Use a single 'user' role message to deliver the entire context, history, and instructions
        full_user_prompt = f"""{system_instruction}
        
Context from documents:
{context_str}

{metrics_str}

{history_str}

Question: {query}

Please provide a helpful answer based on the context and metrics provided."""

        # Create prompt template with only a 'user' role placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("user", "{full_prompt}")
        ])
        
        # Generate messages for LLM invocation
        messages = prompt.format_messages(
            full_prompt=full_user_prompt
        )
        
        try:
            # Invoke the LangChain LLM object
            response = self.llm.invoke(messages)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            # Catches errors from the LLM service (like API key issues, context too long, etc.)
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
