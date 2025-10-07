"""
Gemini AI service for generating predictions and responses.
Fast, reliable AI service with good free tier.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Gemini
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
    logger.info("Google Generative AI imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Google Generative AI: {e}")
    GEMINI_AVAILABLE = False


@dataclass
class GeminiConfig:
    """Configuration for Gemini AI service"""

    model_name: str = "gemini-pro"  # Stable Gemini model
    temperature: float = 0.3  # Lower for faster, more focused responses
    max_tokens: int = 150  # Reduced for faster responses
    api_key: Optional[str] = None


class GeminiService:
    """Service class for interacting with Google Gemini AI models"""

    def __init__(self, config: Optional[GeminiConfig] = None):
        """
        Initialize Gemini service with configuration.

        Args:
            config: GeminiConfig object. If None, will use default configuration.
        """
        self.config = config or GeminiConfig()

        # Get API key from environment or config
        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")

        if not GEMINI_AVAILABLE:
            logger.error("Google Generative AI package not available")
            self.client = None
        elif not api_key:
            logger.warning("No GEMINI_API_KEY found. Service will fail without it.")
            self.client = None
        else:
            try:
                # Configure Gemini
                genai.configure(api_key=api_key)

                # List available models to find what's actually available
                logger.info("Discovering available Gemini models...")
                available_models = []
                try:
                    for model in genai.list_models():
                         if "generateContent" in model.supported_generation_methods:
                            available_models.append(model.name)
                            logger.info(f"Found available model: {model.name}")
                except Exception as list_error:
                    logger.warning(f"Could not list models: {list_error}")

                # Try different model names in order of preference
                model_candidates = [
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-pro",
                    "gemini-1.0-pro",
                ]

                # Add any available models we discovered to the front of the list
                if available_models:
                    model_candidates = available_models + model_candidates

                self.client = None
                successful_model = None

                for model_name in model_candidates:
                    try:
                        logger.info(f"Trying to initialize with model: {model_name}")
                        self.client = genai.GenerativeModel(model_name)

                        # Test the model with a simple generation
                        test_response = self.client.generate_content(
                            "Say 'hello'",
                            generation_config=genai.types.GenerationConfig(
                                max_output_tokens=5
                            ),
                        )

                        if test_response and test_response.text:
                            successful_model = model_name
                            logger.info(
                                f"Successfully initialized and tested Gemini with model: {model_name}"
                            )
                            break
                        else:
                            logger.warning(
                                f"Model {model_name} initialized but test failed"
                            )

                    except Exception as model_error:
                        logger.warning(
                            f"Failed to initialize with model {model_name}: {model_error}"
                        )
                        continue

                if not self.client:
                    logger.error("Failed to initialize with any available model")
                    logger.info(f"Available models from API: {available_models}")
                else:
                    # Update config to use the successful model
                    self.config.model_name = successful_model

            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None

    async def generate_prediction(
        self,
        prompt: str,
        context: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a prediction based on prompt and context.

        Args:
            prompt: The main prompt
            context: Additional context information
            model_name: Override default model name (ignored for Gemini)
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            Generated response text
        """
        try:
            if not self.client:
                raise ValueError(
                    "Gemini client not initialized. Check GEMINI_API_KEY environment variable."
                )

            # Construct the full prompt
            full_prompt = self._construct_prompt(prompt, context)

            logger.info(
                f"Generating prediction with Gemini model: {self.config.model_name}"
            )
            logger.debug(f"Prompt length: {len(full_prompt)} characters")

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature or self.config.temperature,
                max_output_tokens=max_tokens or self.config.max_tokens,
            )

            # Generate content using Gemini
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate_content(
                    full_prompt, generation_config=generation_config
                ),
            )

            # Extract response text
            if not response or not response.text:
                raise ValueError("Empty or invalid response from Gemini")

            prediction_text = response.text.strip()

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

        # Add context if provided
        if context:
            # Add brief instruction for placeholders if context mentions anonymization
            if "anonymized" in context.lower() or "placeholder" in context.lower():
                prompt_parts.append(
                    "Note: Angle brackets like <name> are placeholders, not real data."
                )
            prompt_parts.append(f"Context: {context}")

        # Add the main prompt
        prompt_parts.append(prompt)

        return "\n".join(prompt_parts)

    def validate_connection(self) -> bool:
        """
        Validate that the Gemini service is properly configured and can connect.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Simply check if client is initialized and API key exists
            if not self.client:
                logger.error("Gemini client not initialized")
                return False

            # Check if API key is available
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY environment variable not set")
                return False

            logger.info("Gemini service validation passed")
            return True

        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            return False

    def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in Gemini.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model exists, False otherwise
        """
        available_models = self.list_available_models()
        return model_name in available_models


# Singleton instance for use across the application
_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """
    Get or create a singleton instance of GeminiService.

    Returns:
        GeminiService instance
    """
    global _gemini_service

    if _gemini_service is None:
        _gemini_service = GeminiService()

    return _gemini_service
