"""
Ollama AI service for generating predictions and responses.
Handles local Ollama integration with proper error handling and configuration.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import ollama

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama AI service"""

    model_name: str = "llama3.2:1b"  # Back to working model
    temperature: float = 0.7
    max_tokens: int = 300  # Smaller for speed
    top_p: float = 0.8
    top_k: int = 40
    host: str = "http://localhost:11434"  # Default Ollama host


class OllamaService:
    """Service class for interacting with local Ollama AI models"""

    def __init__(
        self, config: Optional[OllamaConfig] = None, host: Optional[str] = None
    ):
        """
        Initialize Ollama service with configuration.

        Args:
            config: OllamaConfig object. If None, will use default configuration.
            host: Ollama server host. If None, will use default localhost.
        """
        self.config = config or OllamaConfig()

        # Check for environment variable first (for Docker)
        if host:
            self.config.host = host
        elif os.getenv("OLLAMA_HOST"):
            self.config.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Set up the client
        self.client = ollama.Client(host=self.config.host)

        logger.info(f"Ollama service initialized with host: {self.config.host}")

    async def generate_prediction(
        self,
        prompt: str,
        context: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Generate a prediction based on prompt and context.

        Args:
            prompt: The main prompt
            context: Additional context information
            model_name: Override default model name

        Returns:
            Generated response text
        """
        try:
            # Use provided model or default
            model = model_name or self.config.model_name

            # Validate model exists before attempting generation
            try:
                available_models = self.client.list()
                model_names = [m["name"] for m in available_models.get("models", [])]
                if model not in model_names:
                    logger.warning(
                        f"Model {model} not found. Available models: {model_names}"
                    )
                    # Fallback to first available model or llama3.2:1b
                    if "llama3.2:1b" in model_names:
                        model = "llama3.2:1b"
                        logger.info(f"Using fallback model: {model}")
                    elif model_names:
                        model = model_names[0]
                        logger.info(f"Using first available model: {model}")
                    else:
                        raise ValueError("No models available in Ollama")
            except Exception as e:
                logger.warning(
                    f"Could not validate model, proceeding with {model}: {e}"
                )

            # Construct the full prompt
            full_prompt = self._construct_prompt(prompt, context)

            logger.info(f"Generating prediction with Ollama model: {model}")
            logger.debug(f"Prompt length: {len(full_prompt)} characters")

            # Generate content using ollama with performance optimizations
            response = self.client.generate(
                model=model,
                prompt=full_prompt,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": self.config.max_tokens,
                    "num_ctx": 2048,  # Smaller context window for speed
                    "repeat_penalty": 1.1,  # Prevent repetition
                    "seed": -1,  # Random seed for variety
                },
            )

            # Extract response text
            if not response or "response" not in response:
                raise ValueError("Empty or invalid response from Ollama")

            prediction_text = response["response"]

            logger.info("Successfully generated prediction")
            return prediction_text

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise

    def _construct_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Construct the full prompt from components.

        Args:
            prompt: The main prompt
            context: Additional context

        Returns:
            Constructed full prompt
        """
        prompt_parts = []

        # Add system instruction for handling placeholders and anonymized data
        system_instruction = (
            "You are a helpful assistant. When you see placeholders like <name>, <iban1>, <iban2>, "
            "or similar bracketed text, these are NOT real sensitive data - they are placeholders "
            "that should be kept exactly as they are in your response. Treat them as template variables. "
            "Do not refuse to help with requests containing such placeholders."
        )
        prompt_parts.append(f"System: {system_instruction}")

        # Add context if provided
        if context:
            prompt_parts.append(f"Context: {context}")

        # Add the main prompt
        prompt_parts.append(f"User: {prompt}")
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    def list_available_models(self) -> List[str]:
        """
        List available Ollama models.

        Returns:
            List of available model names
        """
        try:
            models = self.client.list()
            return [model["name"] for model in models.get("models", [])]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    def validate_connection(self) -> bool:
        """
        Validate that the Ollama service is properly configured and can connect.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Try to list models as a connection test
            models = self.client.list()
            return bool(models)

        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            return False

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model to local Ollama instance.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model_name}")
            self.client.pull(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False

    def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists locally.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model exists, False otherwise
        """
        available_models = self.list_available_models()
        return model_name in available_models


# Singleton instance for use across the application
_ollama_service: Optional[OllamaService] = None


def get_ollama_service() -> OllamaService:
    """
    Get or create a singleton instance of OllamaService.

    Returns:
        OllamaService instance
    """
    global _ollama_service

    if _ollama_service is None:
        _ollama_service = OllamaService()

    return _ollama_service
