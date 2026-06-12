from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langsmith import traceable

from app.config import get_settings

class AgentConfig(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used:str
    
class ProductionAgent:
    def __init__(self, config: AgentConfig):
        settings = get_settings()
        
        self.primary_llm = ChatGroq(
            model=settings.primary_model,
            temperature=settings.temperature,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
            api_key=settings.api_key,
        )
        self.fallback_llm = ChatGroq(
            model=settings.fallback_model,
            temperature=settings.temperature,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
            api_key=settings.api_key,
        )
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()
        
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
        graph.add_conditional_edge(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"}
        )
        graph.add_conditional_edge(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"}
        )
        
        graph.add_edge("error", END)
        
        return graph.compile()
    
    @traceable(name="production_agent_invoke")
    def invoke(self, messages: str) -> dict:
        
        result = self.graph.run({
            "messages": messages,
            "retry_count": 0,
            "error": None,
            "model_used": "",
        })
        
        return {
            "response": result.get("messages", [])[-1].content,
            "model_used": result.get("model_used","unknown"),
            "error": result.get("error"),
        }