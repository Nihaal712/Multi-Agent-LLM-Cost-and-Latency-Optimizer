"""
Core system logic for LLM Cost & Latency Optimizer.

Implements deterministic execution controller, optimized RAG pipeline,
and cost/latency estimation.
"""

import os
from typing import Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

import config
import utils


def execution_controller(
    complexity_out: Dict[str, Any],
    cost_out: Dict[str, Any],
    latency_out: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministic execution controller that merges agent outputs.
    
    Applies conflict resolution rules to select:
    - model_tier: Selected tier based on SLA preference and budget constraints
    - context_size: Minimum of cost max_tokens and complexity max_context_size
    - retrieval_depth: Minimum of latency max_depth and complexity retrieval_depth
    
    Args:
        complexity_out: Output from query_complexity agent
        cost_out: Output from cost_policy agent
        latency_out: Output from latency_policy agent
    
    Returns:
        Dictionary with:
        - model_tier: str
        - context_size: int
        - retrieval_depth: int
        - full_trace: dict (nested trace combining all agent traces and decisions)
    """
    # Extract outputs from agent results
    complexity_output = complexity_out.get("output", {})
    cost_output = cost_out.get("output", {})
    latency_output = latency_out.get("output", {})
    
    # Get traces
    complexity_trace = complexity_out.get("trace", {})
    cost_trace = cost_out.get("trace", {})
    latency_trace = latency_out.get("trace", {})
    
    # Apply conflict resolution rules (pass output dicts, not full agent results)
    tier_conflict = config.CONFLICT_RULES["resolve_tier_conflict"](
        complexity_output, cost_output, latency_output
    )
    
    context_resolution = config.CONFLICT_RULES["resolve_context_size"](
        complexity_output, cost_output, latency_output
    )
    
    depth_resolution = config.CONFLICT_RULES["resolve_retrieval_depth"](
        complexity_output, cost_output, latency_output
    )
    
    # Select tier: preferred tier if in allowed_tiers, otherwise closest by latency
    preferred_tier = tier_conflict["preferred_tier"]
    allowed_tiers = tier_conflict["allowed_tiers"]
    selected_tier = config.select_tier_from_allowed(preferred_tier, allowed_tiers)
    
    # Determine context_size: min of cost max_tokens and complexity max_context_size
    cost_max_tokens = context_resolution["cost_max_tokens"]
    complexity_max = context_resolution["complexity_max"]
    context_size = min(cost_max_tokens, complexity_max)
    
    # Also cap at tier's max_context_tokens
    tier_max = config.TIER_PROPERTIES[selected_tier].max_context_tokens
    context_size = min(context_size, tier_max)
    
    # Determine retrieval_depth: min of latency max_depth and complexity retrieval_depth
    latency_max_depth = depth_resolution["latency_max_depth"]
    complexity_max_depth = depth_resolution["complexity_max_depth"]
    retrieval_depth = min(latency_max_depth, complexity_max_depth)
    
    # Build full trace
    full_trace = {
        "execution_controller": utils.trace_logger(
            "execution_controller",
            {
                "complexity": complexity_output,
                "cost": cost_output,
                "latency": latency_output,
            },
            {
                "tier_conflict": tier_conflict,
                "context_resolution": context_resolution,
                "depth_resolution": depth_resolution,
                "selected_tier": selected_tier,
                "context_size": context_size,
                "retrieval_depth": retrieval_depth,
            }
        ),
        "agent_traces": {
            "complexity": complexity_trace,
            "cost": cost_trace,
            "latency": latency_trace,
        }
    }
    
    return {
        "model_tier": selected_tier,
        "context_size": context_size,
        "retrieval_depth": retrieval_depth,
        "full_trace": full_trace,
    }


async def optimized_rag(
    query: str,
    decisions: Dict[str, Any]
) -> Tuple[str, int]:
    """
    Optimized RAG pipeline using LCEL chain.
    
    Builds chain: Embed query → FAISS retriever → prompt → LLM
    Enforces context token limits and uses selected tier/model.
    
    Args:
        query: User query string
        decisions: Dictionary from execution_controller with:
            - model_tier: str
            - context_size: int (token limit)
            - retrieval_depth: int (k for retriever)
    
    Returns:
        Tuple of (answer: str, token_usage: int)
    """
    model_tier = decisions["model_tier"]
    context_size = decisions["context_size"]
    retrieval_depth = decisions["retrieval_depth"]
    
    # Ensure environment variables from .env are loaded
    # This is intentionally called here to be robust regardless of entrypoint
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Load corpus (FAISS index)
    vectorstore, metadata_list, texts = utils.load_sample_corpus()
    
    # Create retriever with retrieval_depth as k
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": retrieval_depth}
    )
    
    # Retrieve documents
    retrieved_docs = await retriever.ainvoke(query)
    
    # Format context from retrieved documents
    context_parts = [doc.page_content for doc in retrieved_docs]
    context = "\n\n".join(context_parts)
    
    # Truncate context if it exceeds context_size (in tokens)
    context_tokens = utils.token_count(context)
    if context_tokens > context_size:
        # Simple truncation: keep first N characters that fit
        # More sophisticated: truncate by tokens
        target_chars = int((context_size / context_tokens) * len(context))
        context = context[:target_chars]
        # Re-count after truncation
        context_tokens = utils.token_count(context)
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain enough information,
say so clearly. Be concise and accurate."""),
        ("human", """Context:
{context}

Question: {query}

Answer:"""),
    ])
    
    # Create LLM with selected tier
    model_name = config.MODEL_REGISTRY[model_tier]
    llm = ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=config.DEFAULTS["temperature"],
        max_tokens=min(2048, context_size - context_tokens),  # Reserve space for output
    )
    
    # Build LCEL chain
    # Format inputs: take query string and add context
    def format_inputs(query_str: str) -> dict:
        return {"context": context, "query": query_str}
    
    chain = (
        RunnablePassthrough()
        | format_inputs
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    # Invoke chain with query
    answer = await chain.ainvoke(query)
    
    # Calculate total token usage
    answer_tokens = utils.token_count(answer)
    total_tokens = context_tokens + answer_tokens
    
    return answer, total_tokens


def estimate_cost_latency(
    decisions: Dict[str, Any],
    tokens: int
) -> Dict[str, float]:
    """
    Estimate cost and latency based on decisions and actual token usage.
    
    Uses static TIER_PROPERTIES for calculations.
    
    Args:
        decisions: Dictionary from execution_controller with model_tier
        tokens: Actual token usage from RAG
    
    Returns:
        Dictionary with est_cost (float) and est_latency (float)
    """
    model_tier = decisions["model_tier"]
    tier_props = config.TIER_PROPERTIES[model_tier]
    
    # Calculate cost: (tokens / 1000) * cost_per_1k_tokens
    est_cost = (tokens / 1000.0) * tier_props.cost_per_1k_tokens
    
    # Latency is static per tier (could be enhanced with token-based scaling)
    est_latency = tier_props.latency_seconds
    
    return {
        "est_cost": est_cost,
        "est_latency": est_latency,
    }


# Main guard for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_core():
        print("=== Core.py Sanity Check ===\n")
        
        # Test execution_controller
        print("Testing execution_controller...")
        test_complexity = {
            "output": {"label": "medium", "explanation": "Test"},
            "trace": {"step": "query_complexity"}
        }
        test_cost = {
            "output": {
                "allowed_tiers": ["SMALL", "MEDIUM"],
                "max_tokens": 4096,
                "explanation": "Test"
            },
            "trace": {"step": "cost_policy"}
        }
        test_latency = {
            "output": {
                "preferred_tier": "MEDIUM",
                "max_depth": 5,
                "explanation": "Test"
            },
            "trace": {"step": "latency_policy"}
        }
        
        result = execution_controller(test_complexity, test_cost, test_latency)
        print(f"Selected tier: {result['model_tier']}")
        print(f"Context size: {result['context_size']}")
        print(f"Retrieval depth: {result['retrieval_depth']}")
        print(f"Trace keys: {list(result['full_trace'].keys())}\n")
        
        # Test estimate_cost_latency
        print("Testing estimate_cost_latency...")
        estimates = estimate_cost_latency(result, tokens=2000)
        print(f"Estimated cost: ${estimates['est_cost']:.6f}")
        print(f"Estimated latency: {estimates['est_latency']}s\n")
        
        print("✓ Core.py sanity check complete!")
        print("\nNote: optimized_rag requires GROQ_API_KEY and corpus setup.")
    
    asyncio.run(test_core())

