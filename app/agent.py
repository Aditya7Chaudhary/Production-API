from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langsmith import traceable
import logging

from app.config import get_settings

logger = logging.getLogger("production-api")

class AgentConfig(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    
class AgentState(AgentConfig):
    messages: Annotated[list, add_messages]
    current_task: str
    is_completed: bool
    
class ProductionAgent:
    def __init__(self):
        settings = get_settings()
        
        # 1. FIXED: Initialize the health flag to False by default
        self.is_ready = False 
        
        try:
            self.primary_llm = ChatGroq(
                model=settings.primary_model,
                temperature=settings.temperature,
                timeout=settings.timeout,
                max_retries=settings.max_retries,
                api_key=settings.groq_api_key,
            )
            self.fallback_llm = ChatGroq(
                model=settings.fallback_model,
                temperature=settings.temperature,
                timeout=settings.timeout,
                max_retries=settings.max_retries,
                api_key=settings.groq_api_key,
            )
            self.max_retries = settings.max_retries
            self.graph = self._build_graph()
            
            # 2. FIXED: Flip flag to True once everything compiles and builds successfully!
            self.is_ready = True 
            
        except Exception as e:
            logger.error(f"Failed to initialize ProductionAgent: {e}")
            self.is_ready = False
    
    def check_health(self) -> str:
        """
        Checks if the agent service is initialized and ready to accept tasks.
        Returns "healthy" or "unhealthy".
        """
        try:
            # 3. FIXED: This now runs without throwing an AttributeError
            if self.is_ready and self.graph is not None:
                return "healthy"
            return "unhealthy"
        except Exception as e:
            logger.error(f"Agent health check encountered an error: {e}")
            return "unhealthy"
        
    def _build_graph(self):
        
        def process_messages(state: AgentState) -> dict:
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "retry_count": state.get("retry_count", 0) + 1,
                    "error": str(e),
                    "model_used": "",
                }
            
        def try_fallback(state: AgentState) -> dict:
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }
                
        def handle_error(state: AgentState) -> dict:
            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, but I'm currently experiencing issues processing your request."
                        "Please try again later or contact support if the problem persists."
                    ))
                ],
                "model_used": "error_handler",
            }
            
        def route_after_process(state: AgentState) -> str:
            if state.get("error") is None:
                return "done"
            elif state.get("retry_count", 0) < self.max_retries:
                return "fallback"
            else:
                return "error"
            
        def route_after_fallback(state: AgentState) -> str:
            if state.get("error") is None:
                return "done"
            else:
                return "error"
            
        graph = StateGraph(AgentState)

        graph.add_node("process", process_messages)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)
        
        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"}
        )
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"}
        )
        
        graph.add_edge("error", END)
        
        return graph.compile()
    
    @traceable(name="production_agent_invoke")
    def invoke(self, messages: str) -> dict:
        # Wrap the incoming raw string string into a HumanMessage list
        # because ChatGroq and LangGraph expect structured message streams
        initial_messages = [HumanMessage(content=messages)]
        
        # Run the graph using the correct keyword argument dictionary pattern
        try:
            # Compiled graphs use .invoke() in modern LangGraph
            result = self.graph.invoke({
                "messages": initial_messages,
                "retry_count": 0,
                "error": None,
                "model_used": "initializing",
            })
        except Exception as graph_err:
            logger.error(f"Graph execution fatal crash: {graph_err}")
            return {
                "response": "An internal framework error occurred.",
                "model_used": "graph_system_failure",
                "error": str(graph_err)
            }
        
        # Securely extract messages safely
        final_messages = result.get("messages", [])
        response_content = "No response generated."
        if final_messages:
            response_content = final_messages[-1].content

        return {
            "response": response_content,
            "model_used": result.get("model_used", "unknown"), # Will now correctly pull 'primary' or 'fallback'
            "error": result.get("error"),
        }