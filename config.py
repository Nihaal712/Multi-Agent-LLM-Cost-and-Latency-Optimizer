"""
Configuration module for LLM Cost & Latency Optimizer.

Defines model registry, tier properties, defaults, SLA preferences,
complexity defaults, and conflict resolution rules.
"""

from typing import Dict, List, Callable, Any
from dataclasses import dataclass

# Model Registry: Maps tier names to Groq model identifiers
MODEL_REGISTRY: Dict[str, str] = {
    "SMALL": "llama-3.1-8b-instant",  # Fast, lower cost
    "MEDIUM": "llama-3.1-8b-instant",  # Temporarily use SMALL model
    "LARGE": "llama-3.1-8b-instant",  # Temporarily use SMALL model
}

# Tier Properties: Static cost (per 1K tokens), latency (seconds), and context size
@dataclass
class TierProperties:
    """Properties for each model tier."""
    cost_per_1k_tokens: float  # USD per 1K tokens
    latency_seconds: float  # Estimated latency in seconds
    max_context_tokens: int  # Maximum context window size

TIER_PROPERTIES: Dict[str, TierProperties] = {
    "SMALL": TierProperties(
        cost_per_1k_tokens=0.0001,  # Lower cost estimate
        latency_seconds=0.5,  # Fast response
        max_context_tokens=8192,
    ),
    "MEDIUM": TierProperties(
        cost_per_1k_tokens=0.0005,  # Medium cost
        latency_seconds=1.5,  # Balanced latency
        max_context_tokens=8192,
    ),
    "LARGE": TierProperties(
        cost_per_1k_tokens=0.001,  # Higher cost
        latency_seconds=3.0,  # Slower but more capable
        max_context_tokens=8192,
    ),
}

# Defaults
DEFAULTS: Dict[str, Any] = {
    "budget": None,  # No budget limit by default
    "sla": "balanced",  # Default SLA preference
    "default_tier": "MEDIUM",  # Fallback tier
    "default_context_size": 4096,  # Default context tokens
    "default_retrieval_depth": 5,  # Default number of docs to retrieve
    "temperature": 0,  # Deterministic (no randomness)
}

# SLA to Tier Preference Mapping
SLA_TO_TIER_PREF: Dict[str, str] = {
    "fast": "SMALL",  # Fast SLA prefers small, fast models
    "balanced": "MEDIUM",  # Balanced SLA prefers medium tier
    "relaxed": "LARGE",  # Relaxed SLA can use large, capable models
}

# Complexity Defaults
COMPLEXITY_DEFAULTS: Dict[str, Any] = {
    "simple": {
        "preferred_tier": "SMALL",
        "max_context_size": 2048,
        "retrieval_depth": 3,
    },
    "medium": {
        "preferred_tier": "MEDIUM",
        "max_context_size": 4096,
        "retrieval_depth": 5,
    },
    "complex": {
        "preferred_tier": "LARGE",
        "max_context_size": 8192,
        "retrieval_depth": 10,
    },
}

# Conflict Resolution Rules (deterministic lambdas)
# These functions resolve conflicts between cost policy and latency policy outputs
CONFLICT_RULES: Dict[str, Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    # Rule 1: SLA preference takes precedence over budget if preferred tier is not in allowed_tiers
    # If preferred tier is not allowed, pick the closest allowed tier by latency
    "resolve_tier_conflict": lambda complexity_out, cost_out, latency_out: {
        "logic": "SLA preference prioritized; if not in allowed_tiers, select closest by latency",
        "preferred_tier": latency_out.get("preferred_tier", DEFAULTS["default_tier"]),
        "allowed_tiers": cost_out.get("allowed_tiers", list(MODEL_REGISTRY.keys())),
    },
    
    # Rule 2: Context size = min(max_tokens from cost, complexity-based context)
    "resolve_context_size": lambda complexity_out, cost_out, latency_out: {
        "logic": "Context size = min(cost max_tokens, complexity max_context)",
        "cost_max_tokens": cost_out.get("max_tokens", DEFAULTS["default_context_size"]),
        "complexity_max": COMPLEXITY_DEFAULTS.get(
            complexity_out.get("label", "medium"), {}
        ).get("max_context_size", DEFAULTS["default_context_size"]),
    },
    
    # Rule 3: Retrieval depth = min(latency max_depth, complexity-based depth)
    "resolve_retrieval_depth": lambda complexity_out, cost_out, latency_out: {
        "logic": "Retrieval depth = min(latency max_depth, complexity max_depth)",
        "latency_max_depth": latency_out.get("max_depth", DEFAULTS["default_retrieval_depth"]),
        "complexity_max_depth": COMPLEXITY_DEFAULTS.get(
            complexity_out.get("label", "medium"), {}
        ).get("retrieval_depth", DEFAULTS["default_retrieval_depth"]),
    },
}

# Helper function to select tier when preferred is not in allowed list
def select_tier_from_allowed(preferred_tier: str, allowed_tiers: List[str]) -> str:
    """
    Select the best tier from allowed_tiers, preferring the one closest to preferred_tier.
    Falls back to first allowed tier if preferred is not available.
    """
    if preferred_tier in allowed_tiers:
        return preferred_tier
    
    # Tier priority order (by latency: SMALL < MEDIUM < LARGE)
    tier_order = ["SMALL", "MEDIUM", "LARGE"]
    
    # Find preferred index
    try:
        preferred_idx = tier_order.index(preferred_tier)
    except ValueError:
        preferred_idx = 1  # Default to MEDIUM
    
    # Find closest allowed tier by latency
    best_tier = None
    best_distance = float('inf')
    
    for tier in allowed_tiers:
        if tier in tier_order:
            distance = abs(tier_order.index(tier) - preferred_idx)
            if distance < best_distance:
                best_distance = distance
                best_tier = tier
    
    return best_tier or allowed_tiers[0] if allowed_tiers else DEFAULTS["default_tier"]


# Main guard for testing
if __name__ == "__main__":
    print("=== Config.py Sanity Check ===\n")
    
    print("MODEL_REGISTRY:")
    for tier, model in MODEL_REGISTRY.items():
        print(f"  {tier}: {model}")
    
    print("\nTIER_PROPERTIES:")
    for tier, props in TIER_PROPERTIES.items():
        print(f"  {tier}:")
        print(f"    Cost per 1K tokens: ${props.cost_per_1k_tokens}")
        print(f"    Latency: {props.latency_seconds}s")
        print(f"    Max context: {props.max_context_tokens} tokens")
    
    print("\nDEFAULTS:")
    for key, value in DEFAULTS.items():
        print(f"  {key}: {value}")
    
    print("\nSLA_TO_TIER_PREF:")
    for sla, tier in SLA_TO_TIER_PREF.items():
        print(f"  {sla}: {tier}")
    
    print("\nCOMPLEXITY_DEFAULTS:")
    for complexity, defaults in COMPLEXITY_DEFAULTS.items():
        print(f"  {complexity}: {defaults}")
    
    print("\nCONFLICT_RULES:")
    for rule_name in CONFLICT_RULES.keys():
        print(f"  {rule_name}: defined")
    
    # Test conflict resolution
    print("\n=== Testing Conflict Resolution ===")
    test_complexity = {"label": "medium", "explanation": "Test"}
    test_cost = {"allowed_tiers": ["SMALL", "MEDIUM"], "max_tokens": 2048, "explanation": "Budget limited"}
    test_latency = {"preferred_tier": "LARGE", "max_depth": 7, "explanation": "Relaxed SLA"}
    
    tier_conflict = CONFLICT_RULES["resolve_tier_conflict"](test_complexity, test_cost, test_latency)
    print(f"Tier conflict resolution: {tier_conflict}")
    
    selected_tier = select_tier_from_allowed(tier_conflict["preferred_tier"], tier_conflict["allowed_tiers"])
    print(f"Selected tier: {selected_tier}")
    
    context_resolution = CONFLICT_RULES["resolve_context_size"](test_complexity, test_cost, test_latency)
    print(f"Context size resolution: {context_resolution}")
    
    depth_resolution = CONFLICT_RULES["resolve_retrieval_depth"](test_complexity, test_cost, test_latency)
    print(f"Retrieval depth resolution: {depth_resolution}")
    
    print("\n✓ Config.py sanity check complete!")

