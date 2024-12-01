from typing import List, Dict, Any
import numpy as np
import json
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from embeddings import cosine_similarity, embed_texts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
AGENT_LOOP_LIMIT = 3

class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) engine that combines OpenAI's language models
    with a local knowledge base for context-aware responses.
    """
    
    def __init__(self) -> None:
        """Initialize the RAG engine with OpenAI client and configuration."""
        try:
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Configuration
        self.model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
        self.agent_loop_limit = AGENT_LOOP_LIMIT
        
        # Sample data - in production this would come from a database
        self.data = [
            "Python is a versatile programming language used for web development, data analysis, and more.",
            "OpenAI provides advanced AI models like GPT-4 that support function calling.",
            "Function calling allows external tools to be integrated seamlessly into chatbots.",
            "Machine learning is a subset of artificial intelligence that focuses on building algorithms.",
            "The Turing test is a benchmark for evaluating an AI's ability to mimic human intelligence.",
            "Transformers are a type of neural network architecture that powers modern AI systems.",
            "Kotlin is a modern programming language, widely used for Android app development.",
            "Docker and Kubernetes are essential tools for containerized application deployment.",
        ]
        
        # Tool definitions
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_context",
                    "description": "Retrieve relevant context from the dataset based on the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's query to find relevant context."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # Response schema
        self.context_reasoning = {
            "type": "json_schema",
            "json_schema": {
                "name": "context_reasoning",
                "schema": {
                "type": "object",
                "properties": {
                    "reasoning": { "type": "string" },
                    "final_answer": { "type": "string" }
                },
                "required": ["reasoning", "final_answer"],
                "additionalProperties": False
                },
                "strict": True
            }
        }
        
        # Initial system message
        self.messages = [
           {
                "role": "system",
                "content": (
                    "You are an AI assistant whose primary goal is to answer user questions effectively. "
                    "When a user's question lacks sufficient information, use the `retrieve_context` tool to find relevant information. "
                    "If retrieving additional context doesn't help, ask the user to clarify their question for more details. "
                    "Avoid excessive looping to find answers if the information is unavailable; instead, be transparent and admit if you don't know."
                )
            }
        ]

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve the most relevant context for the given query using embedding similarity.
        
        Args:
            query (str): The user's query to find relevant context for
            
        Returns:
            str: The most relevant context from the knowledge base
        """
        try:
            data_embeddings = embed_texts(self.data)
            query_embedding = embed_texts([query])[0].reshape(1, -1)
            
            similarities = [
                cosine_similarity(query_embedding, data_embedding.reshape(1, -1)) 
                for data_embedding in data_embeddings
            ]
            
            most_relevant_idx = np.argmax(similarities)
            return self.data[most_relevant_idx]
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def get_response(self, query: str) -> Dict[str, Any]:
        """
        Get a structured reasoning response for the given query.
        
        Args:
            query (str): The user's question
            
        Returns:
            Dict[str, Any]: A dictionary containing intermediate steps, reasoning, and final answer
        """
        self.messages.append({"role": "user", "content": query})
        try:
            intermediate_steps = []
            # Initial function call to retrieve context
            initial_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                response_format=self.context_reasoning
            )
            
            loop_response = initial_response
            count = 0
            while loop_response.choices[0].message.tool_calls and count < self.agent_loop_limit:
                # Execute all tool calls
                tool_call_results_message = []
                for tool_call in loop_response.choices[0].message.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    # Add tool input step
                    intermediate_steps.append({
                        "explanation": "Tool Input",
                        "output": f"Function: {tool_call.function.name}, Arguments: {json.dumps(arguments)}"
                    })
                    
                    context = self.retrieve_context(arguments.get("query", query))
                    tool_call_results_message.append({
                        "role": "tool",
                        "content": context,
                        "tool_call_id": tool_call.id
                    })
                    
                    # Add tool response step
                    intermediate_steps.append({
                        "explanation": "Tool Response",
                        "output": context
                    })
                
                # Update messages with context and reasoning instruction
                self.messages.extend([
                    loop_response.choices[0].message,
                    *tool_call_results_message
                ])

                # Final call for tool response and reasoning
                loop_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    response_format=self.context_reasoning
                )
                count += 1
            
            final_response = json.loads(loop_response.choices[0].message.content) if loop_response.choices[0].message.content is not None else {"reasoning": "Stuck in loop", "final_answer": "Error: Stuck in loop"}
            # Append assistant response
            self.messages.append({"role": "assistant", "content": final_response["final_answer"]})
            return {
                "intermediate_steps": intermediate_steps if intermediate_steps else [],
                "reasoning": final_response["reasoning"],
                "final_answer": final_response["final_answer"]
            }
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                "intermediate_steps": [{
                    "explanation": "Error",
                    "output": str(e)
                }],
                "reasoning": f"Error occurred: {str(e)}",
                "final_answer": f"Error: {str(e)}"
            }

if __name__ == "__main__":
    engine = RAGEngine()
    response = engine.get_response("What is machine learning?")
    print(json.dumps(response, indent=2))
