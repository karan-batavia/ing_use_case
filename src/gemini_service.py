"""
Gemini AI service for generating predictions and responses.
Handles Google Generative AI integration with proper error handling and configuration.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Configuration for Gemini AI service"""

    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.8
    top_k: int = 40


class GeminiService:
    """Service class for interacting with Google's Gemini AI API"""

    def __init__(
        self, api_key: Optional[str] = None, config: Optional[GeminiConfig] = None
    ):
        """
        Initialize Gemini service with API key and configuration.

        Args:
            api_key: Google API key. If None, will try to get from environment variable.
            config: GeminiConfig object. If None, will use default configuration.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.config = config or GeminiConfig()
        self.model = None

        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Gemini AI client"""
        try:
            genai.configure(api_key=self.api_key)

            # Configure safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=self.config.model_name,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                ),
            )

            logger.info(
                f"Gemini service initialized with model: {self.config.model_name}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise

    def generate_prediction(
        self,
        sanitized_prompt: str,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a prediction based on sanitized prompt and context.

        Args:
            sanitized_prompt: The main prompt that has been sanitized/scrubbed
            context: Additional context information
            system_instruction: System-level instruction for the model

        Returns:
            Dictionary containing the prediction response and metadata
        """
        if not self.model:
            raise RuntimeError("Gemini client not initialized")

        try:
            # Construct the full prompt
            full_prompt = self._construct_prompt(
                sanitized_prompt, context, system_instruction
            )

            logger.info("Generating prediction with Gemini AI")
            logger.debug(f"Prompt length: {len(full_prompt)} characters")

            # Generate content
            response = self.model.generate_content(full_prompt)

            # Extract and validate response
            if not response.text:
                raise ValueError("Empty response from Gemini AI")

            result = {
                "prediction": response.text,
                "model_used": self.config.model_name,
                "prompt_token_count": (
                    response.usage_metadata.prompt_token_count
                    if response.usage_metadata
                    else None
                ),
                "candidates_token_count": (
                    response.usage_metadata.candidates_token_count
                    if response.usage_metadata
                    else None
                ),
                "total_token_count": (
                    response.usage_metadata.total_token_count
                    if response.usage_metadata
                    else None
                ),
                "finish_reason": (
                    response.candidates[0].finish_reason.name
                    if response.candidates
                    else None
                ),
                "safety_ratings": (
                    [
                        {
                            "category": rating.category.name,
                            "probability": rating.probability.name,
                        }
                        for rating in response.candidates[0].safety_ratings
                    ]
                    if response.candidates and response.candidates[0].safety_ratings
                    else []
                ),
            }

            logger.info("Successfully generated prediction")
            return result

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise

    def _construct_prompt(
        self,
        sanitized_prompt: str,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Construct the full prompt from components.

        Args:
            sanitized_prompt: The main sanitized prompt
            context: Additional context
            system_instruction: System instruction

        Returns:
            Constructed full prompt
        """
        prompt_parts = []

        # Add system instruction if provided
        if system_instruction:
            prompt_parts.append(f"System: {system_instruction}")

        # Add context if provided
        if context:
            prompt_parts.append(f"Context: {context}")

        # Add the main prompt
        prompt_parts.append(f"Prompt: {sanitized_prompt}")

        return "\n\n".join(prompt_parts)

    def list_available_models(self) -> List[str]:
        """
        List available Gemini models.

        Returns:
            List of available model names
        """
        try:
            models = genai.list_models()
            return [
                model.name
                for model in models
                if "generateContent" in model.supported_generation_methods
            ]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    def validate_connection(self) -> bool:
        """
        Validate that the Gemini service is properly configured and can connect.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            if not self.model:
                return False

            # Try a simple test generation
            test_response = self.model.generate_content("Test connection")
            return bool(test_response and test_response.text)

        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            return False


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
