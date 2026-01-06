"""
Agent logic for LLM Cost & Latency Optimizer.

Implements LLM-based classification agents for query complexity,
cost policy, and latency policy using LCEL chains with Pydantic output parsing.
"""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

import config
import utils
from dotenv import load_dotenv
load_dotenv()


# Helper function for cost policy calculations
def _calculate_affordable_tokens_per_tier(budget: float = None) -> Dict[str, int]:
    """
    Calculate maximum affordable tokens per tier based on budget.
    Returns dict with tier -> affordable_tokens (capped at 8192).
    
    Args:
        budget: Budget in USD (None = unlimited)
    
    Returns:
        Dictionary mapping tier names to affordable token counts
    """
    if budget is None:
        # Unlimited budget - all tiers can use max context
        return {
            "SMALL": 8192,
            "MEDIUM": 8192,
            "LARGE": 8192,
        }
    
    affordable_tokens = {}
    for tier, props in config.TIER_PROPERTIES.items():
        # Calculate: (budget / cost_per_1k_tokens) * 1000
        # Cap at model's max_context_tokens (8192)
        calculated_tokens = int((budget / props.cost_per_1k_tokens) * 1000)
        affordable_tokens[tier] = min(calculated_tokens, props.max_context_tokens)
    
    return affordable_tokens


# Pydantic Models for Agent Outputs
class ComplexityOutput(BaseModel):
    """Output model for query complexity classification."""
    label: str = Field(description="Complexity label: 'simple', 'medium', or 'complex'")
    explanation: str = Field(description="Brief explanation of the complexity classification")


class CostPolicyOutput(BaseModel):
    """Output model for cost policy decisions."""
    allowed_tiers: list[str] = Field(description="List of allowed model tiers based on budget constraints")
    max_tokens: int = Field(description="Maximum tokens allowed based on budget")
    explanation: str = Field(description="Explanation of cost policy decisions")


class LatencyPolicyOutput(BaseModel):
    """Output model for latency policy decisions."""
    preferred_tier: str = Field(description="Preferred model tier based on SLA requirements")
    max_depth: int = Field(description="Maximum retrieval depth (1-10) based on SLA and complexity")
    explanation: str = Field(description="Explanation of latency policy decisions")


# Query Complexity Agent
def _get_complexity_prompt() -> ChatPromptTemplate:
    """Get the few-shot prompt for query complexity classification."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert at classifying query complexity for LLM systems.
Classify queries into one of three categories:
- "simple": Factual lookups, single-step questions, straightforward information retrieval
- "medium": Requires reasoning, multi-step thinking, or moderate analysis
- "complex": Multi-step reasoning, deep analysis, creative synthesis, or complex problem-solving

Respond ONLY with valid JSON matching this schema:
{{
    "label": "simple|medium|complex",
    "explanation": "brief explanation"
}}"""),
        ("human", "Query: What is the capital of France?\n\nResponse:"),
        ("assistant", '{{"label": "simple", "explanation": "Direct factual lookup question requiring no reasoning."}}'),
        ("human", "Query: Compare the advantages and disadvantages of quantum computing versus classical computing.\n\nResponse:"),
        ("assistant", '{{"label": "medium", "explanation": "Requires multi-step reasoning to compare two computing paradigms."}}'),
        ("human", "Query: Design a machine learning pipeline that can detect anomalies in real-time sensor data while maintaining privacy and explainability.\n\nResponse:"),
        ("assistant", '{{"label": "complex", "explanation": "Multi-faceted problem requiring synthesis of ML, privacy, and explainability concepts."}}'),
        ("human", "Query: {query}\n\nResponse:"),
    ])


async def query_complexity(query: str) -> Dict[str, Any]:
    """
    Classify query complexity using LLM agent.
    
    Args:
        query: User query string
    
    Returns:
        Dictionary with "output" (ComplexityOutput dict) and "trace" (dict)
    """
    trace = utils.trace_logger(
        "query_complexity",
        {"query": query},
        {}
    )
    
    try:
        # Get API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Create LLM with deterministic settings
        llm = ChatGroq(
            model=config.MODEL_REGISTRY["SMALL"],  # Use SMALL tier for classification
            groq_api_key=api_key,
            temperature=config.DEFAULTS["temperature"],
        )
        
        # Create parser
        parser = PydanticOutputParser(pydantic_object=ComplexityOutput)
        
        # Build LCEL chain
        prompt = _get_complexity_prompt()
        chain = prompt | llm | parser
        
        # Invoke chain
        result = await chain.ainvoke({"query": query})
        
        # Convert Pydantic model to dict
        output_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        
        trace["outputs"]["output"] = output_dict
        trace["outputs"]["success"] = True
        
    except Exception as e:
        # Deterministic fallback
        trace["outputs"]["error"] = str(e)
        trace["outputs"]["success"] = False
        output_dict = {
            "label": "medium",  # Default fallback
            "explanation": f"Classification failed, defaulting to medium complexity. Error: {str(e)}"
        }
        trace["outputs"]["output"] = output_dict
    
    return {
        "output": output_dict,
        "trace": trace
    }


# Cost Policy Agent
def _get_cost_policy_prompt() -> ChatPromptTemplate:
    """Get the few-shot prompt for cost policy decisions."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a cost optimization expert for LLM systems.
Given query complexity, budget constraints, and pre-calculated affordable tokens per tier, determine:
1. Which model tiers are allowed (SMALL, MEDIUM, LARGE)
2. Maximum tokens to use (select from affordable tokens of allowed tiers)

IMPORTANT: All models have max_context_tokens=8192. Use the affordable_tokens values provided - they are already calculated and capped at 8192.

Rules:
- A tier is allowed if its affordable_tokens >= 1000 (minimum viable)
- Select allowed_tiers based on which tiers have sufficient affordable tokens
- Set max_tokens to the minimum affordable_tokens value among allowed tiers (or the highest if all are equal)

If budget is None or unlimited, all tiers are allowed with max_tokens=8192.

Respond ONLY with valid JSON matching this schema:
{{
    "allowed_tiers": ["SMALL", "MEDIUM", "LARGE"],
    "max_tokens": 8192,
    "explanation": "brief explanation"
}}"""),
        ("human", """Complexity: simple
Budget: $0.01
Affordable tokens per tier:
- SMALL: 8192 tokens
- MEDIUM: 8192 tokens
- LARGE: 8192 tokens
Response:"""),
        ("assistant", '{{"allowed_tiers": ["SMALL", "MEDIUM", "LARGE"], "max_tokens": 8192, "explanation": "Budget $0.01 allows all tiers. All tiers have 8192 affordable tokens (capped at model limit). All tiers allowed, max_tokens = 8192."}}'),
        ("human", """Complexity: complex
Budget: None
Affordable tokens per tier:
- SMALL: 8192 tokens
- MEDIUM: 8192 tokens
- LARGE: 8192 tokens
Response:"""),
        ("assistant", '{{"allowed_tiers": ["SMALL", "MEDIUM", "LARGE"], "max_tokens": 8192, "explanation": "No budget constraint. All tiers allowed with 8192 affordable tokens. Max_tokens = 8192."}}'),
        ("human", """Complexity: medium
Budget: $0.0005
Affordable tokens per tier:
- SMALL: 5000 tokens
- MEDIUM: 1000 tokens
- LARGE: 500 tokens
Response:"""),
        ("assistant", '{{"allowed_tiers": ["SMALL", "MEDIUM"], "max_tokens": 1000, "explanation": "Budget $0.0005 allows SMALL (5000 tokens) and MEDIUM (1000 tokens). LARGE (500 tokens) is below minimum. Using minimum of allowed tiers: max_tokens = 1000."}}'),
        ("human", """Complexity: simple
Budget: $0.0002
Affordable tokens per tier:
- SMALL: 2000 tokens
- MEDIUM: 400 tokens
- LARGE: 200 tokens
Response:"""),
        ("assistant", '{{"allowed_tiers": ["SMALL"], "max_tokens": 2000, "explanation": "Budget $0.0002 allows only SMALL tier (2000 tokens). MEDIUM (400) and LARGE (200) are below minimum 1000 tokens. Max_tokens = 2000."}}'),
        ("human", """Complexity: {complexity}
Budget: {budget}
Affordable tokens per tier:
{affordable_tokens}
Response:"""),
    ])


async def cost_policy(complexity: Dict[str, Any], budget: float = None) -> Dict[str, Any]:
    """
    Determine cost policy based on complexity and budget.
    
    Args:
        complexity: Complexity output dict with "label" and "explanation"
        budget: Optional budget constraint in USD (None = unlimited)
    
    Returns:
        Dictionary with "output" (CostPolicyOutput dict) and "trace" (dict)
    """
    trace = utils.trace_logger(
        "cost_policy",
        {"complexity": complexity, "budget": budget},
        {}
    )
    
    try:
        # Get API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Create LLM
        llm = ChatGroq(
            model=config.MODEL_REGISTRY["SMALL"],
            groq_api_key=api_key,
            temperature=config.DEFAULTS["temperature"],
        )
        
        # Create parser
        parser = PydanticOutputParser(pydantic_object=CostPolicyOutput)
        
        # Build LCEL chain
        prompt = _get_cost_policy_prompt()
        chain = prompt | llm | parser
        
        # Calculate affordable tokens per tier (deterministic)
        affordable_tokens = _calculate_affordable_tokens_per_tier(budget)
        trace["outputs"]["affordable_tokens"] = affordable_tokens  # For debugging
        
        # Format affordable tokens as string for prompt
        affordable_tokens_str = "\n".join([
            f"- {tier}: {tokens} tokens"
            for tier, tokens in affordable_tokens.items()
        ])
        
        # Format inputs
        complexity_label = complexity.get("output", {}).get("label", "medium")
        budget_str = str(budget) if budget is not None else "None"
        
        # Invoke chain
        result = await chain.ainvoke({
            "complexity": complexity_label,
            "budget": budget_str,
            "affordable_tokens": affordable_tokens_str
        })
        
        # Convert to dict
        output_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        # SAFETY: ensure max_tokens is always positive
        if output_dict.get("max_tokens", 0) <= 0:
            output_dict["max_tokens"] = max(
                1,
                min(affordable_tokens.values()) if affordable_tokens else 1
            )
        trace["outputs"]["output"] = output_dict
        trace["outputs"]["success"] = True
        
    except Exception as e:
        # Deterministic fallback using calculated affordable tokens
        trace["outputs"]["error"] = str(e)
        trace["outputs"]["success"] = False
        
        # Calculate affordable tokens for fallback
        affordable_tokens = _calculate_affordable_tokens_per_tier(budget)
        
        # Determine allowed tiers (minimum 1000 tokens)
        allowed_tiers = [
            tier for tier, tokens in affordable_tokens.items()
            if tokens >= 1000
        ]
        
        if not allowed_tiers:
            # If no tier meets minimum, use SMALL as fallback
            allowed_tiers = ["SMALL"]
            max_tokens = max(1000, affordable_tokens.get("SMALL", 1000))
        else:
            # Use minimum affordable tokens among allowed tiers
            max_tokens = min(affordable_tokens[tier] for tier in allowed_tiers)
        
        output_dict = {
            "allowed_tiers": allowed_tiers,
            "max_tokens": max_tokens,
            "explanation": f"Fallback: Using calculated affordable tokens. Error: {str(e)}"
        }
        trace["outputs"]["output"] = output_dict
    
    return {
        "output": output_dict,
        "trace": trace
    }


# Latency Policy Agent
def _get_latency_policy_prompt() -> ChatPromptTemplate:
    """Get the few-shot prompt for latency policy decisions."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a latency optimization expert for LLM systems.
Given query complexity and SLA requirements, determine:
1. Preferred model tier (SMALL=fast, MEDIUM=balanced, LARGE=relaxed)
2. Maximum retrieval depth (1-10, where higher depth = more context but slower)

SLA mappings:
- "fast": Prefer SMALL tier, lower depth (1-3)
- "balanced": Prefer MEDIUM tier, moderate depth (3-7)
- "relaxed": Prefer LARGE tier, higher depth (5-10)

Complexity affects depth:
- simple: Lower depth (1-5)
- medium: Moderate depth (3-7)
- complex: Higher depth (5-10)

Respond ONLY with valid JSON matching this schema:
{{
    "preferred_tier": "SMALL|MEDIUM|LARGE",
    "max_depth": 5,
    "explanation": "brief explanation"
}}"""),
        ("human", """Complexity: simple
SLA: fast
Response:"""),
        ("assistant", '{{"preferred_tier": "SMALL", "max_depth": 3, "explanation": "Fast SLA with simple query: SMALL tier for speed, depth 3 for quick retrieval."}}'),
        ("human", """Complexity: complex
SLA: relaxed
Response:"""),
        ("assistant", '{{"preferred_tier": "LARGE", "max_depth": 10, "explanation": "Relaxed SLA with complex query: LARGE tier for capability, depth 10 for comprehensive context."}}'),
        ("human", """Complexity: medium
SLA: balanced
Response:"""),
        ("assistant", '{{"preferred_tier": "MEDIUM", "max_depth": 5, "explanation": "Balanced SLA with medium complexity: MEDIUM tier for balance, depth 5 for adequate context."}}'),
        ("human", """Complexity: {complexity}
SLA: {sla}
Response:"""),
    ])


async def latency_policy(complexity: Dict[str, Any], sla: str = "balanced") -> Dict[str, Any]:
    """
    Determine latency policy based on complexity and SLA.
    
    Args:
        complexity: Complexity output dict with "label" and "explanation"
        sla: SLA requirement ("fast", "balanced", or "relaxed")
    
    Returns:
        Dictionary with "output" (LatencyPolicyOutput dict) and "trace" (dict)
    """
    trace = utils.trace_logger(
        "latency_policy",
        {"complexity": complexity, "sla": sla},
        {}
    )
    
    try:
        # Get API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Create LLM
        llm = ChatGroq(
            model=config.MODEL_REGISTRY["SMALL"],
            groq_api_key=api_key,
            temperature=config.DEFAULTS["temperature"],
        )
        
        # Create parser
        parser = PydanticOutputParser(pydantic_object=LatencyPolicyOutput)
        
        # Build LCEL chain
        prompt = _get_latency_policy_prompt()
        chain = prompt | llm | parser
        
        # Format inputs
        complexity_label = complexity.get("output", {}).get("label", "medium")
        sla_normalized = sla.lower() if sla else "balanced"
        
        # Invoke chain
        result = await chain.ainvoke({
            "complexity": complexity_label,
            "sla": sla_normalized
        })
        
        # Convert to dict
        output_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        
        trace["outputs"]["output"] = output_dict
        trace["outputs"]["success"] = True
        
    except Exception as e:
        # Deterministic fallback
        trace["outputs"]["error"] = str(e)
        trace["outputs"]["success"] = False
        
        # Fallback: use config defaults
        preferred_tier = config.SLA_TO_TIER_PREF.get(sla.lower() if sla else "balanced", config.DEFAULTS["default_tier"])
        complexity_label = complexity.get("output", {}).get("label", "medium")
        default_depth = config.COMPLEXITY_DEFAULTS.get(complexity_label, {}).get("retrieval_depth", config.DEFAULTS["default_retrieval_depth"])
        
        output_dict = {
            "preferred_tier": preferred_tier,
            "max_depth": default_depth,
            "explanation": f"Fallback: Using config defaults. Error: {str(e)}"
        }
        trace["outputs"]["output"] = output_dict
    
    return {
        "output": output_dict,
        "trace": trace
    }


# Main guard for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_agents():
        print("=== Agents.py Sanity Check ===\n")
        
        # Test query_complexity
        print("Testing query_complexity...")
        test_query = "What is artificial intelligence?"
        result = await query_complexity(test_query)
        print(f"Query: {test_query}")
        print(f"Output: {result['output']}")
        print(f"Success: {result['trace']['outputs'].get('success', False)}\n")
        
        # Test cost_policy
        print("Testing cost_policy...")
        complexity = {"label": "medium", "explanation": "Test"}
        result = await cost_policy(complexity, budget=0.01)
        print(f"Complexity: {complexity}, Budget: 0.01")
        print(f"Output: {result['output']}")
        print(f"Success: {result['trace']['outputs'].get('success', False)}\n")
        
        # Test latency_policy
        print("Testing latency_policy...")
        result = await latency_policy(complexity, sla="fast")
        print(f"Complexity: {complexity}, SLA: fast")
        print(f"Output: {result['output']}")
        print(f"Success: {result['trace']['outputs'].get('success', False)}\n")
        
        print("✓ Agents.py sanity check complete!")
    
    # Note: This requires GROQ_API_KEY to be set
    # asyncio.run(test_agents())

