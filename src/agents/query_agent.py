"""
Query Agent with Optimized LangGraph

Implements Stage 5 specification with all three tools:
1. pageindex_navigate - Tree traversal for document structure
2. semantic_search - Vector retrieval for content
3. structured_query - SQL over extracted fact tables

Production features:
1. Max iterations limit (prevents infinite loops)
2. Timeout handling (prevents hangs)
3. Optimized prompts (faster responses)
4. Better error handling
5. Performance monitoring

All answers include full provenance: document_name, page_number, bbox, content_hash
"""

import json
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from src.models.types import PageIndex, ProvenanceChain, BBox
from src.agents.indexer import PageIndexQuery
from src.agents.vector_store import VectorStoreManager
from src.agents.fact_extractor import EnhancedFactTableExtractor
from src.llm_factory import get_llm


# Prompt file paths
PROMPTS_DIR = Path(__file__).parent / "prompts"
QUERY_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "query_agent_system.txt"
AUDIT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "query_agent_audit.txt"
SYNTHESIS_PROMPT_PATH = PROMPTS_DIR / "query_agent_synthesis.txt"


def load_prompt(path: Path) -> str:
    """Load prompt template from file"""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()


class AgentState(TypedDict):
    """State for the query agent"""
    messages: List[BaseMessage]
    doc_id: str
    page_index: Optional[PageIndex]
    provenance: List[ProvenanceChain]
    context: List[Dict[str, Any]]
    mode: str  # "query" or "audit"
    iterations: int  # Track iterations


class QueryAgent:
    """
    Query Agent with LangGraph orchestration and three tools
    
    Implements Stage 5 specification with full provenance tracking
    """
    
    def __init__(
        self,
        doc_id: str,
        page_index_path: str,
        llm_provider: str = "openrouter",
        llm_model: str = "arcee-ai/trinity-large-preview:free",  # Fast free model with tool support
        max_iterations: int = 100  # Increased from 3 to allow more tool calls
    ):
        self.doc_id = doc_id
        self.page_index_path = page_index_path
        self.max_iterations = max_iterations
        
        # Load PageIndex
        with open(page_index_path, 'r') as f:
            data = json.load(f)
        self.page_index = PageIndex.model_validate(data)
        
        # Initialize components
        self.page_index_query = PageIndexQuery(self.page_index)
        self.vector_store = VectorStoreManager()
        self.fact_extractor = EnhancedFactTableExtractor()
        
        # Initialize LLM with faster model
        self.llm = get_llm(provider=llm_provider, model=llm_model)
        
        # Build tools
        self.tools = self._build_tools()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_tools(self):
        """Build the three core tools as per Stage 5 specification"""
        
        @tool
        def pageindex_navigate(query: str, top_k: int = 3) -> str:
            """
            Navigate the document structure using PageIndex tree traversal.
            Returns relevant sections with page ranges.
            
            Args:
                query: The search query
                top_k: Number of top sections to return (default: 3, max: 5)
            """
            # Limit top_k for speed
            top_k = min(top_k, 5)
            
            results = self.page_index_query.query(query, top_k=top_k)
            
            sections = []
            for section, score in results:
                sections.append({
                    "title": section.title,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "summary": section.summary[:200],  # Truncate for speed
                    "score": float(score)
                })
            
            return json.dumps(sections, indent=2)
        
        @tool
        def semantic_search(query: str, top_k: int = 3, include_images: bool = False) -> str:
            """
            Search document content using vector similarity.
            Returns relevant chunks with page numbers and bounding boxes.
            
            Args:
                query: The search query
                top_k: Number of results (default: 3, max: 5)
                include_images: If True, include base64-encoded images for vision models (default: False)
            """
            # Limit top_k for speed
            top_k = min(top_k, 5)
            
            results = self.vector_store.search(self.doc_id, query, k=top_k)
            
            search_results = []
            for content, score, metadata in results:
                result = {
                    "content": content[:300],  # Truncate for speed
                    "page": metadata.get("page_min", -1),
                    "hash": metadata.get("content_hash", "")[:16],
                    "bbox": metadata.get("bbox"),  # Include bbox for spatial traceability
                    "chunk_type": metadata.get("chunk_type", "text")  # Include chunk type
                }
                
                # If this is a figure chunk, include image path and caption
                if metadata.get("chunk_type") == "figure":
                    image_path = metadata.get("image_path")
                    result["image_path"] = image_path
                    result["caption"] = metadata.get("caption")
                    
                    # Optionally include base64 image data for vision models
                    if include_images and image_path:
                        try:
                            import base64
                            from pathlib import Path
                            if Path(image_path).exists():
                                with open(image_path, 'rb') as img_file:
                                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                    # Only include if image is reasonably small (< 500KB)
                                    if len(img_data) < 500000:
                                        result["image_base64"] = img_data[:1000] + "..." if len(img_data) > 1000 else img_data
                                        result["image_note"] = "Image available but truncated for context size"
                        except Exception as e:
                            result["image_error"] = f"Could not load image: {str(e)}"
                
                search_results.append(result)
            
            return json.dumps(search_results, indent=2)
        
        @tool
        def structured_query(entity_search: str) -> str:
            """
            Query the fact table for structured data (SQL over extracted facts).
            Use for financial numbers, dates, percentages.
            
            Args:
                entity_search: Entity to search for (e.g., "interest income")
            """
            results = self.fact_extractor.query_facts(
                entity_query=entity_search,
                doc_id=self.doc_id,
                limit=3
            )
            
            facts = []
            for r in results:
                facts.append({
                    "entity": r['entity'],
                    "value": r['value'],
                    "page": r['page_num']
                })
            
            return json.dumps(facts, indent=2)
        
        # All three tools as per Stage 5 specification
        return [pageindex_navigate, semantic_search, structured_query]
    
    def _build_graph(self):
        """Build optimized LangGraph workflow"""
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        def should_continue(state: AgentState):
            """Decide whether to continue or end"""
            messages = state["messages"]
            last_message = messages[-1] if messages else None
            iterations = state.get("iterations", 0)
            
            # If there are no tool calls, we're done (LLM has given final answer)
            if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return "end"
            
            # If we've hit max iterations, force one final answer generation
            if iterations >= self.max_iterations:
                return "final_answer"
            
            return "continue"
        
        def generate_final_answer(state: AgentState):
            """Force LLM to generate final answer without tools"""
            messages = state["messages"]
            
            # Extract tool results from messages to create a summary
            tool_results = []
            for msg in messages:
                if hasattr(msg, "content") and hasattr(msg, "name"):
                    # This is a tool message
                    tool_name = msg.name if hasattr(msg, "name") else "unknown_tool"
                    tool_content = msg.content[:800] if hasattr(msg, "content") else ""
                    tool_results.append(f"=== Tool: {tool_name} ===\n{tool_content}")
            
            # Load synthesis prompt template
            synthesis_template = load_prompt(SYNTHESIS_PROMPT_PATH)
            
            # Format the prompt with tool results
            synthesis_prompt = synthesis_template.format(
                tool_results="\n".join(tool_results[-3:]) if tool_results else "No tool results available"
            )
            
            final_instruction = HumanMessage(content=synthesis_prompt)
            
            # Use LLM without tools for final answer
            response = self.llm.invoke(messages + [final_instruction])
            
            return {
                "messages": messages + [final_instruction, response]
            }
        
        def call_model(state: AgentState):
            """Call the LLM with tools"""
            messages = state["messages"]
            iterations = state.get("iterations", 0)
            
            # Add system message if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                system_msg = SystemMessage(content=self._get_system_prompt(state["mode"]))
                messages = [system_msg] + messages
            
            response = llm_with_tools.invoke(messages)
            
            return {
                "messages": messages + [response],
                "iterations": iterations + 1
            }
        
        def call_tools(state: AgentState):
            """Execute tool calls"""
            tool_node = ToolNode(self.tools)
            return tool_node.invoke(state)
        
        def extract_provenance(state: AgentState):
            """Extract provenance from tool results"""
            provenance_list = state.get("provenance", [])
            
            # Look for provenance in the last tool results
            for message in reversed(state["messages"]):
                if hasattr(message, "content") and isinstance(message.content, str):
                    try:
                        data = json.loads(message.content)
                        
                        # Handle list of results
                        if isinstance(data, list):
                            for item in data:
                                if "page" in item and "hash" in item:
                                    # Parse bbox if present
                                    bbox_data = item.get("bbox")
                                    bbox = None
                                    if bbox_data and isinstance(bbox_data, dict):
                                        try:
                                            bbox = BBox(**bbox_data)
                                        except Exception:
                                            # If bbox parsing fails, continue without it
                                            pass
                                    
                                    provenance_list.append(ProvenanceChain(
                                        document_name=self.doc_id,
                                        page_number=item.get("page", 1),
                                        bbox=bbox,  # Now includes actual bbox
                                        content_hash=item.get("hash", "")
                                    ))
                    except:
                        pass
            
            return {"provenance": provenance_list}
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)
        workflow.add_node("final_answer", generate_final_answer)
        workflow.add_node("extract_provenance", extract_provenance)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "final_answer": "final_answer",
                "end": "extract_provenance"
            }
        )
        
        # Tools always go back to agent
        workflow.add_edge("tools", "agent")
        
        # Final answer goes to provenance extraction
        workflow.add_edge("final_answer", "extract_provenance")
        
        # Provenance extraction ends
        workflow.add_edge("extract_provenance", END)
        
        return workflow.compile()
    
    def _get_system_prompt(self, mode: str) -> str:
        """Get optimized system prompt from external file"""

        if mode == "audit":
            template = load_prompt(AUDIT_SYSTEM_PROMPT_PATH)
        else:
            template = load_prompt(QUERY_SYSTEM_PROMPT_PATH)

        # Format the template with instance variables
        return template.format(
            doc_id=self.doc_id,
            max_iterations=self.max_iterations
        )
    
    def audit(self, claim: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Audit mode with timeout
        
        Args:
            claim: Claim to verify
            timeout: Timeout in seconds (default: 30)
        
        Returns:
            Dict with verification status and provenance
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Audit timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            initial_state = {
                "messages": [HumanMessage(content=f"VERIFY: {claim}")],
                "doc_id": self.doc_id,
                "page_index": self.page_index,
                "provenance": [],
                "context": [],
                "mode": "audit",
                "iterations": 0
            }
            
            result = self.graph.invoke(initial_state)
            
            # Cancel timeout
            signal.alarm(0)
            
            # Extract verification result
            final_message = result["messages"][-1]
            verification = final_message.content if hasattr(final_message, "content") else str(final_message)
            
            # Determine status
            status = "VERIFIED" if "VERIFIED" in verification.upper() else "NOT_FOUND"
            
            return {
                "claim": claim,
                "status": status,
                "verification": verification,
                "provenance": [p.model_dump() for p in result.get("provenance", [])],
                "doc_id": self.doc_id,
                "iterations": result.get("iterations", 0)
            }
            
        except TimeoutError:
            signal.alarm(0)
            return {
                "claim": claim,
                "status": "TIMEOUT",
                "verification": f"Audit timed out after {timeout} seconds",
                "provenance": [],
                "doc_id": self.doc_id,
                "iterations": self.max_iterations
            }
        except Exception as e:
            signal.alarm(0)
            return {
                "claim": claim,
                "status": "ERROR",
                "verification": f"Error: {str(e)}",
                "provenance": [],
                "doc_id": self.doc_id,
                "iterations": 0
            }
    
    def query(self, question: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Query mode with timeout
        
        Args:
            question: User question
            timeout: Timeout in seconds (default: 30)
        
        Returns:
            Dict with answer and provenance
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Query timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "doc_id": self.doc_id,
                "page_index": self.page_index,
                "provenance": [],
                "context": [],
                "mode": "query",
                "iterations": 0
            }
            
            result = self.graph.invoke(initial_state)
            
            # Cancel timeout
            signal.alarm(0)
            
            # Extract final answer - look for the last AIMessage without tool calls
            answer = ""
            for msg in reversed(result["messages"]):
                # Skip system messages and tool messages
                if isinstance(msg, (SystemMessage, HumanMessage)):
                    continue
                    
                # Look for AIMessage with content but no tool calls
                if isinstance(msg, AIMessage):
                    if hasattr(msg, "content") and msg.content:
                        # Skip if this message has tool calls (it's not the final answer)
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            continue
                        answer = msg.content
                        break
                # Fallback: check for any message with content attribute
                elif hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                    answer = msg.content
                    break
            
            # If still no answer, provide a helpful fallback
            if not answer or answer.strip() == "":
                answer = "No answer was generated. The LLM may not support tool calling properly. Try using a different model (e.g., Groq or GPT-4)."
            
            return {
                "answer": answer,
                "provenance": [p.model_dump() for p in result.get("provenance", [])],
                "doc_id": self.doc_id,
                "iterations": result.get("iterations", 0),
                "messages": [str(m)[:200] for m in result.get("messages", [])]  # Debug info
            }
            
        except TimeoutError:
            signal.alarm(0)
            return {
                "answer": f"Query timed out after {timeout} seconds",
                "provenance": [],
                "doc_id": self.doc_id,
                "iterations": self.max_iterations
            }
        except Exception as e:
            signal.alarm(0)
            return {
                "answer": f"Error: {str(e)}",
                "provenance": [],
                "doc_id": self.doc_id,
                "iterations": 0
            }


def main():
    """Example usage"""
    
    print("="*80)
    print("PRODUCTION QUERY AGENT - OPTIMIZED")
    print("="*80)
    print()
    
    # Initialize agent
    agent = QueryAgent(
        doc_id="CBE ANNUAL REPORT 2023-24.pdf",
        page_index_path=".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json",
        max_iterations=3  # Limit to 3 iterations
    )
    
    # Test audit
    print("Testing audit mode...")
    result = agent.audit("The interest income was 101,040,098,062 ETB in 2024", timeout=30)
    
    print(f"\nClaim: {result['claim']}")
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Verification: {result['verification']}")
    print()


if __name__ == "__main__":
    main()
