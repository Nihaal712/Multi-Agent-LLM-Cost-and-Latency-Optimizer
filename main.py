"""
Main orchestration module for LLM Cost & Latency Optimizer.

Implements LangGraph-based workflow for query processing with cost and latency optimization.
"""

import argparse
import asyncio
from typing import TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, START, END

import agents
import core
import config


# Define State as TypedDict
class State(TypedDict):
    """State schema for the orchestration graph."""
    query: str
    budget: Optional[float]
    sla: Optional[str]
    complexity: Dict[str, Any]
    cost: Dict[str, Any]
    latency: Dict[str, Any]
    decisions: Dict[str, Any]
    answer: str
    tokens: int
    trace: Dict[str, Any]
    estimates: Dict[str, float]


# Node functions
def init_state(state: State) -> Dict[str, Any]:
    """
    Initialize state with defaults for missing values.
    
    Args:
        state: Current state
    
    Returns:
        Partial state with initialized defaults
    """
    updates = {}
    
    # Set default SLA if not provided
    if state.get("sla") is None:
        updates["sla"] = config.DEFAULTS["sla"]
    
    # Initialize empty dicts for agent outputs if not present
    if "complexity" not in state:
        updates["complexity"] = {}
    if "cost" not in state:
        updates["cost"] = {}
    if "latency" not in state:
        updates["latency"] = {}
    if "decisions" not in state:
        updates["decisions"] = {}
    if "trace" not in state:
        updates["trace"] = {}
    if "estimates" not in state:
        updates["estimates"] = {}
    
    # Initialize answer and tokens
    if "answer" not in state:
        updates["answer"] = ""
    if "tokens" not in state:
        updates["tokens"] = 0
    
    return updates


async def run_complexity_agent(state: State) -> Dict[str, Any]:
    """
    Run the complexity classification agent.
    
    Args:
        state: Current state with query
    
    Returns:
        Partial state with complexity output
    """
    query = state["query"]
    result = await agents.query_complexity(query)
    
    return {
        "complexity": result,
        "trace": {
            **state.get("trace", {}),
            "complexity_agent": result.get("trace", {})
        }
    }


async def run_parallel_policies(state: State) -> Dict[str, Any]:
    """
    Run cost and latency policy agents in parallel.
    
    Args:
        state: Current state with complexity output
    
    Returns:
        Partial state with cost and latency outputs
    """
    complexity = state["complexity"]
    budget = state.get("budget")
    sla = state.get("sla", config.DEFAULTS["sla"])
    
    # Run both policies in parallel
    cost_result, latency_result = await asyncio.gather(
        agents.cost_policy(complexity, budget),
        agents.latency_policy(complexity, sla)
    )
    
    return {
        "cost": cost_result,
        "latency": latency_result,
        "trace": {
            **state.get("trace", {}),
            "cost_agent": cost_result.get("trace", {}),
            "latency_agent": latency_result.get("trace", {})
        }
    }


def aggregate_controller(state: State) -> Dict[str, Any]:
    """
    Aggregate agent outputs using execution controller.
    
    Args:
        state: Current state with complexity, cost, and latency outputs
    
    Returns:
        Partial state with decisions
    """
    complexity_out = state["complexity"]
    cost_out = state["cost"]
    latency_out = state["latency"]
    
    decisions = core.execution_controller(
        complexity_out,
        cost_out,
        latency_out
    )
    
    return {
        "decisions": decisions,
        "trace": {
            **state.get("trace", {}),
            "execution_controller": decisions.get("full_trace", {})
        }
    }


async def run_rag(state: State) -> Dict[str, Any]:
    """
    Run optimized RAG pipeline.
    
    Args:
        state: Current state with decisions
    
    Returns:
        Partial state with answer and tokens
    """
    query = state["query"]
    decisions = state["decisions"]
    
    answer, tokens = await core.optimized_rag(query, decisions)
    
    return {
        "answer": answer,
        "tokens": tokens
    }


def estimate(state: State) -> Dict[str, Any]:
    """
    Estimate cost and latency based on decisions and token usage.
    
    Args:
        state: Current state with decisions and tokens
    
    Returns:
        Partial state with estimates
    """
    decisions = state["decisions"]
    tokens = state["tokens"]
    
    estimates = core.estimate_cost_latency(decisions, tokens)
    
    return {
        "estimates": estimates
    }


# Build the graph
def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create StateGraph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("init_state", init_state)
    graph.add_node("run_complexity_agent", run_complexity_agent)
    graph.add_node("run_parallel_policies", run_parallel_policies)
    graph.add_node("aggregate_controller", aggregate_controller)
    graph.add_node("run_rag", run_rag)
    graph.add_node("estimate", estimate)
    
    # Add edges
    graph.add_edge(START, "init_state")
    graph.add_edge("init_state", "run_complexity_agent")
    graph.add_edge("run_complexity_agent", "run_parallel_policies")
    graph.add_edge("run_parallel_policies", "aggregate_controller")
    graph.add_edge("aggregate_controller", "run_rag")
    graph.add_edge("run_rag", "estimate")
    graph.add_edge("estimate", END)
    
    # Compile graph
    return graph.compile()


# Global compiled graph instance
_compiled_graph = None


def get_graph():
    """Get or create the compiled graph instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def orchestrate(
    query: str,
    budget: Optional[float] = None,
    sla: Optional[str] = None
) -> State:
    """
    Orchestrate the complete query processing workflow.
    
    Args:
        query: User query string
        budget: Optional budget constraint in USD
        sla: Optional SLA requirement ("fast", "balanced", or "relaxed")
    
    Returns:
        Final state with answer, decisions, estimates, and trace
    """
    # Get compiled graph
    graph = get_graph()
    
    # Initial state
    initial_state: State = {
        "query": query,
        "budget": budget,
        "sla": sla,
        "complexity": {},
        "cost": {},
        "latency": {},
        "decisions": {},
        "answer": "",
        "tokens": 0,
        "trace": {},
        "estimates": {}
    }
    
    # Run graph
    final_state = await graph.ainvoke(initial_state)
    
    return final_state


# CLI integration
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Cost & Latency Optimizer - Query processing with optimization"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query string to process"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Optional budget constraint in USD (e.g., 0.01)"
    )
    parser.add_argument(
        "--sla",
        type=str,
        choices=["fast", "balanced", "relaxed"],
        default=None,
        help="Optional SLA requirement: fast, balanced, or relaxed"
    )
    return parser.parse_args()


def print_results(state: State):
    """Print formatted results from orchestration."""
    print("\n" + "=" * 80)
    print("LLM Cost & Latency Optimizer - Results")
    print("=" * 80)
    
    print(f"\n📝 Query: {state['query']}")
    
    if state.get("budget"):
        print(f"💰 Budget: ${state['budget']:.6f}")
    if state.get("sla"):
        print(f"⏱️  SLA: {state['sla']}")
    
    print("\n" + "-" * 80)
    print("📊 Complexity Analysis")
    print("-" * 80)
    complexity_output = state.get("complexity", {}).get("output", {})
    print(f"  Label: {complexity_output.get('label', 'N/A')}")
    print(f"  Explanation: {complexity_output.get('explanation', 'N/A')}")
    
    print("\n" + "-" * 80)
    print("🎯 Execution Decisions")
    print("-" * 80)
    decisions = state.get("decisions", {})
    print(f"  Model Tier: {decisions.get('model_tier', 'N/A')}")
    print(f"  Context Size: {decisions.get('context_size', 'N/A')} tokens")
    print(f"  Retrieval Depth: {decisions.get('retrieval_depth', 'N/A')}")
    
    print("\n" + "-" * 80)
    print("💡 Answer")
    print("-" * 80)
    print(f"  {state.get('answer', 'N/A')}")
    
    print("\n" + "-" * 80)
    print("📈 Metrics")
    print("-" * 80)
    print(f"  Tokens Used: {state.get('tokens', 0)}")
    estimates = state.get("estimates", {})
    print(f"  Estimated Cost: ${estimates.get('est_cost', 0):.6f}")
    print(f"  Estimated Latency: {estimates.get('est_latency', 0):.2f}s")
    
    print("\n" + "=" * 80)
    print("✓ Processing complete!")
    print("=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Run orchestration
    try:
        final_state = asyncio.run(orchestrate(
            query=args.query,
            budget=args.budget,
            sla=args.sla
        ))
        
        # Print results
        print_results(final_state)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

