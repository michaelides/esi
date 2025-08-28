"""
OpenRouter Model Manager

This module provides a robust interface for managing OpenRouter models,
including validation, configuration, and error handling.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)


class OpenRouterModelManager:
    """Manages OpenRouter model configurations and creates LLM instances."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the model manager with configuration.
        
        Args:
            config_path: Path to the model configuration JSON file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "openrouter_models.json"
        
        self.config_path = Path(config_path)
        self.model_registry = self._load_model_registry()
        
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry from configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Model registry file not found: {self.config_path}")
            return {"models": {}, "providers": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in model registry: {e}")
            return {"models": {}, "providers": {}}
    
    def validate_model(self, model_id: str) -> bool:
        """Check if a model ID is supported.
        
        Args:
            model_id: The model identifier to validate
            
        Returns:
            True if model is supported, False otherwise
        """
        return model_id in self.model_registry.get("models", {})
    
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ValueError: If model is not supported
        """
        if not self.validate_model(model_id):
            available_models = list(self.model_registry.get("models", {}).keys())
            raise ValueError(
                f"Unsupported OpenRouter model: '{model_id}'. "
                f"Available models: {available_models}"
            )
        
        return self.model_registry["models"][model_id]
    
    def get_max_temperature_for_model(self, model_id: str) -> float:
        """Get the maximum temperature supported by a model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Maximum temperature value
        """
        try:
            config = self.get_model_config(model_id)
            return config.get("max_temperature", 2.0)
        except ValueError:
            # Default for unknown models
            return 2.0
    
    def constrain_temperature(self, temperature: float, model_id: str) -> float:
        """Constrain temperature to valid range for the model.
        
        Args:
            temperature: Requested temperature value
            model_id: The model identifier
            
        Returns:
            Constrained temperature value
        """
        max_temp = self.get_max_temperature_for_model(model_id)
        return max(0.0, min(temperature, max_temp))
    
    def create_llm(self, model_id: str, temperature: float, **kwargs) -> ChatOpenAI:
        """Create a ChatOpenAI instance for OpenRouter.
        
        Args:
            model_id: The model identifier
            temperature: Temperature parameter
            **kwargs: Additional parameters for ChatOpenAI
            
        Returns:
            Configured ChatOpenAI instance
            
        Raises:
            ValueError: If model is not supported or API key is missing
        """
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for OpenRouter models"
            )
        
        config = self.get_model_config(model_id)
        constrained_temp = self.constrain_temperature(temperature, model_id)
        
        # Get model-specific kwargs
        model_kwargs = config.get("model_kwargs", {})
        
        # Merge with provided kwargs, giving precedence to provided ones
        final_kwargs = {**model_kwargs, **kwargs}
        
        llm_params = {
            "model": model_id,
            "temperature": constrained_temp,
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "request_timeout": 30,
            "max_retries": 3,
            **final_kwargs
        }
        
        logger.info(f"Creating OpenRouter LLM for model: {model_id}")
        logger.debug(f"LLM parameters: {llm_params}")
        
        return ChatOpenAI(**llm_params)
    
    def get_supported_models(self) -> List[str]:
        """Get list of all supported model IDs.
        
        Returns:
            List of supported model identifiers
        """
        return list(self.model_registry.get("models", {}).keys())
    
    def get_models_by_provider(self, provider: str) -> List[str]:
        """Get models filtered by provider.
        
        Args:
            provider: Provider name (e.g., "OpenAI", "Anthropic")
            
        Returns:
            List of model IDs from the specified provider
        """
        models = []
        for model_id, config in self.model_registry.get("models", {}).items():
            if config.get("provider") == provider:
                models.append(model_id)
        return models
    
    def get_models_by_pricing_tier(self, tier: str) -> List[str]:
        """Get models filtered by pricing tier.
        
        Args:
            tier: Pricing tier ("free", "standard", "premium")
            
        Returns:
            List of model IDs in the specified pricing tier
        """
        models = []
        for model_id, config in self.model_registry.get("models", {}).items():
            if config.get("pricing_tier") == tier:
                models.append(model_id)
        return models
    
    def handle_api_error(self, error: Exception, model_id: str) -> str:
        """Handle OpenRouter API errors with user-friendly messages.
        
        Args:
            error: The exception that occurred
            model_id: The model that caused the error
            
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        # Common error patterns and their user-friendly messages
        error_mappings = {
            "invalid_model": f"Model '{model_id}' is not available or has been deprecated",
            "quota": "API quota exceeded. Please try again later or upgrade your plan",
            "rate": "Rate limit exceeded. Please wait a moment before trying again",
            "timeout": "Request timed out. Please check your connection and try again",
            "unauthorized": "Invalid API key. Please check your OpenRouter API key configuration",
            "insufficient": "Insufficient credits. Please add credits to your OpenRouter account",
            "context": f"Input too long for model '{model_id}'. Please reduce the input size",
            "network": "Network connectivity issue. Please check your internet connection"
        }
        
        # Check for specific error patterns
        for pattern, message in error_mappings.items():
            if pattern in error_str:
                logger.warning(f"OpenRouter API error ({pattern}): {error}")
                return message
        
        # Generic error message
        logger.error(f"Unexpected OpenRouter API error: {error}")
        return f"OpenRouter API error: {str(error)}"


# Global instance for convenience
_manager_instance = None

def get_openrouter_manager() -> OpenRouterModelManager:
    """Get a singleton instance of the OpenRouter model manager.
    
    Returns:
        OpenRouterModelManager instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = OpenRouterModelManager()
    return _manager_instance


def is_openrouter_model(model_id: str) -> bool:
    """Check if a model ID is an OpenRouter model.
    
    Args:
        model_id: Model identifier to check
        
    Returns:
        True if it's an OpenRouter model, False otherwise
    """
    if not model_id:
        return False
    
    # Gemini models are handled separately
    if model_id.startswith("gemini"):
        return False
    
    # Mistral models are handled separately
    if model_id.startswith("mistral"):
        return False
    
    # Check if model is in our registry
    manager = get_openrouter_manager()
    return manager.validate_model(model_id)


def constrain_temperature_for_model(temperature: float, model_id: str) -> float:
    """Constrain temperature based on model type.
    
    Args:
        temperature: Requested temperature
        model_id: Model identifier
        
    Returns:
        Constrained temperature value
    """
    if model_id and model_id.startswith("gemini"):
        # Gemini models only accept temperature 0.0-1.0
        return max(0.0, min(temperature, 1.0))
    elif is_openrouter_model(model_id):
        # Use model-specific constraints
        manager = get_openrouter_manager()
        return manager.constrain_temperature(temperature, model_id)
    else:
        # Default for unknown models
        return max(0.0, min(temperature, 2.0))