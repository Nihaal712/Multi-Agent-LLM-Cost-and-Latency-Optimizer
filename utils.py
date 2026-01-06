"""
Utility functions for LLM Cost & Latency Optimizer.

Includes LLM calls, token counting, trace logging, and corpus loading.
"""

import os
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime , timezone
import tiktoken

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
from dotenv import load_dotenv
load_dotenv()

async def llm_call(
    prompt: str,
    tier: str = "SMALL",
    retries: int = 3,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Async LLM call using ChatGroq with retry logic.
    
    Args:
        prompt: The input prompt string
        tier: Model tier (SMALL/MEDIUM/LARGE) - defaults to SMALL
        retries: Number of retry attempts on failure
        temperature: Temperature for generation (default 0 for deterministic)
        max_tokens: Maximum tokens to generate (optional)
    
    Returns:
        Generated text response as string
    
    Raises:
        Exception: If all retries fail
    """
    # Get model name from registry
    model_name = config.MODEL_REGISTRY.get(tier, config.MODEL_REGISTRY["SMALL"])
    
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Initialize ChatGroq with deterministic settings
    llm = ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Retry logic
    last_error = None
    for attempt in range(retries):
        try:
            response = await llm.ainvoke(prompt)
            # Extract text from response
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
            else:
                raise Exception(f"LLM call failed after {retries} attempts: {str(last_error)}")
    
    raise Exception(f"LLM call failed: {str(last_error)}")


def token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text to count tokens for
        model: Model name for tokenizer (default: gpt-3.5-turbo)
    
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def trace_logger(step: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a trace log entry with timestamp and structured data.
    
    Args:
        step: Name of the step/operation
        inputs: Input data dictionary
        outputs: Output data dictionary
    
    Returns:
        Dictionary with step name, timestamp, inputs, and outputs
    """
    return {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": inputs,
        "outputs": outputs,
    }


def load_sample_corpus() -> Tuple[FAISS, List[Dict[str, Any]], List[str]]:
    """
    Load or create sample corpus with embeddings and FAISS index.
    
    Generates sample documents if they don't exist, then:
    1. Loads text files from data/sample_docs/
    2. Embeds them using HuggingFaceEmbeddings
    3. Builds FAISS vector store
    4. Returns index, metadata list, and texts list
    
    Returns:
        Tuple of (FAISS index, metadata list, texts list)
    """
    # Ensure data directory exists
    data_dir = "data/sample_docs"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate sample documents if they don't exist
    _generate_sample_documents(data_dir)
    
    # Load documents
    texts = []
    metadata_list = []
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                    metadata_list.append({
                        "source": filename,
                        "filepath": filepath,
                    })
    
    if not texts:
        raise ValueError("No sample documents found in data/sample_docs/")
    
    # Initialize embeddings (using a lightweight model for MVP)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # Use CPU for MVP
    )
    
    # Create FAISS index
    # Convert texts to Document objects for FAISS
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadata_list)
    ]
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore, metadata_list, texts


def _generate_sample_documents(data_dir: str) -> None:
    """
    Generate sample documents if they don't exist.
    Creates 3-5 sample .txt files with AI/quantum themes, 200-500 words each.
    """
    sample_docs = {
        "ai_faq.txt": """Artificial Intelligence (AI) represents one of the most transformative technologies of our time. At its core, AI involves creating computer systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine learning, a subset of AI, enables systems to learn and improve from experience without being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has revolutionized fields like image recognition, natural language processing, and autonomous vehicles.

Large Language Models (LLMs) like GPT and LLaMA have demonstrated remarkable capabilities in understanding and generating human-like text. These models are trained on vast amounts of text data and can perform tasks ranging from translation to creative writing.

AI applications span numerous domains including healthcare, where it aids in diagnosis and drug discovery; finance, for fraud detection and algorithmic trading; and transportation, powering autonomous vehicles. However, AI also raises important ethical considerations around bias, privacy, job displacement, and the need for responsible development.

The future of AI holds promise for solving complex global challenges while requiring careful governance to ensure beneficial outcomes for humanity.""",

        "quantum_basics.txt": """Quantum computing represents a paradigm shift from classical computing, leveraging the principles of quantum mechanics to process information in fundamentally different ways. Unlike classical bits that exist in states of 0 or 1, quantum bits or qubits can exist in superpositions, allowing them to be in multiple states simultaneously.

This property of superposition, combined with entanglement—where qubits become correlated in ways that cannot be explained classically—enables quantum computers to perform certain calculations exponentially faster than classical computers. Quantum algorithms like Shor's algorithm for factoring and Grover's algorithm for search demonstrate this potential.

Key challenges in quantum computing include maintaining quantum coherence, which is fragile and easily disrupted by environmental noise. Quantum error correction and fault-tolerant quantum computing are active areas of research to address these challenges.

Current quantum computers are in the NISQ (Noisy Intermediate-Scale Quantum) era, with devices containing dozens to hundreds of qubits. Applications being explored include cryptography, optimization problems, drug discovery, and materials science.

Major technology companies and research institutions are investing heavily in quantum computing, with various approaches including superconducting qubits, trapped ions, and photonic quantum systems. The race toward practical quantum advantage continues.""",

        "llm_optimization.txt": """Large Language Model optimization involves techniques to improve efficiency, reduce costs, and enhance performance of LLM deployments. Key strategies include model quantization, which reduces precision of model weights to decrease memory and computational requirements while maintaining acceptable accuracy.

Prompt engineering is crucial for optimizing LLM interactions, involving careful crafting of prompts to elicit desired responses with minimal tokens. Few-shot learning and chain-of-thought prompting can improve model performance on complex tasks.

Retrieval-Augmented Generation (RAG) combines LLMs with external knowledge bases, allowing models to access up-to-date information without retraining. This approach reduces hallucination and enables domain-specific applications.

Cost optimization strategies include selecting appropriate model sizes for tasks, implementing caching for repeated queries, and using batch processing. Latency optimization involves model pruning, knowledge distillation, and efficient inference frameworks.

Token management is critical, as costs scale with input and output tokens. Techniques like context window management, response length limits, and smart truncation help control expenses while maintaining quality.""",

        "neural_networks.txt": """Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes or neurons organized in layers: input, hidden, and output layers. Each connection has an associated weight that gets adjusted during training.

The training process involves forward propagation, where input data flows through the network, and backpropagation, where errors are propagated backward to adjust weights. This iterative process minimizes the difference between predicted and actual outputs.

Deep neural networks with many hidden layers can learn hierarchical representations, with early layers detecting simple features and deeper layers combining them into complex patterns. Convolutional Neural Networks (CNNs) excel at image processing, while Recurrent Neural Networks (RNNs) and Transformers are powerful for sequential data.

Activation functions like ReLU, sigmoid, and tanh introduce non-linearity, enabling networks to learn complex relationships. Regularization techniques like dropout and batch normalization prevent overfitting and improve generalization.

Modern neural networks power applications from computer vision to natural language processing, with architectures continuously evolving to improve efficiency and performance.""",

        "ai_ethics.txt": """AI ethics addresses the moral principles and values that should guide the development and deployment of artificial intelligence systems. Key concerns include algorithmic bias, where AI systems may perpetuate or amplify existing societal inequalities due to biased training data or design choices.

Transparency and explainability are crucial for building trust in AI systems, especially in high-stakes applications like healthcare and criminal justice. The "black box" nature of many AI models makes it difficult to understand their decision-making processes.

Privacy concerns arise from AI systems that collect and process vast amounts of personal data. Data protection regulations like GDPR aim to safeguard individual privacy rights while enabling beneficial AI applications.

Job displacement is a significant concern as AI automation transforms industries. However, AI also creates new job categories and can augment human capabilities rather than replace them entirely.

Responsible AI development requires diverse teams, ethical guidelines, and ongoing monitoring. Organizations must consider the broader societal impacts of their AI systems and work toward equitable, beneficial outcomes for all stakeholders."""
    }
    
    # Generate files if they don't exist
    for filename, content in sample_docs.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)


# Main guard for testing
if __name__ == "__main__":
    print("=== Utils.py Sanity Check ===\n")
    
    # Test token_count
    test_text = "This is a test sentence for token counting."
    token_num = token_count(test_text)
    print(f"Token count test: '{test_text}' = {token_num} tokens")
    
    # Test trace_logger
    test_trace = trace_logger(
        "test_step",
        {"input": "test_input"},
        {"output": "test_output"}
    )
    print(f"\nTrace logger test: {test_trace}")
    
    # Test load_sample_corpus (this will generate documents if needed)
    print("\nLoading sample corpus...")
    try:
        vectorstore, metadata, texts = load_sample_corpus()
        print(f"✓ Loaded {len(texts)} documents")
        print(f"✓ Created FAISS index with {len(metadata)} entries")
        print(f"✓ Sample metadata: {metadata[0] if metadata else 'None'}")
        print(f"✓ First document preview: {texts[0][:100]}...")
    except Exception as e:
        print(f"✗ Error loading corpus: {e}")
    
    print("\n✓ Utils.py sanity check complete!")

