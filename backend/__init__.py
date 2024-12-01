"""
Agentic RAG Chat Backend Package

This package implements an Agentic RAG (Retrieval-Augmented Generation) system
with OpenAI integration, providing context-aware responses with step-by-step reasoning.
"""

from .embeddings import cosine_similarity, embed_texts
from .rag_engine import RAGEngine, Step, ContextReasoning

__all__ = [
    'cosine_similarity',
    'embed_texts',
    'RAGEngine',
    'Step',
    'ContextReasoning'
]
