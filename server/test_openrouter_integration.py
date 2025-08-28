"""
Test suite for OpenRouter model integration

This module provides comprehensive tests for the OpenRouter model manager
and integration with the agent system.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock
from openrouter_manager import (
    OpenRouterModelManager, 
    get_openrouter_manager, 
    is_openrouter_model, 
    constrain_temperature_for_model
)
from agent import create_agent


class TestOpenRouterModelManager:
    """Test cases for the OpenRouterModelManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a test model manager instance."""
        return OpenRouterModelManager()
    
    def test_load_model_registry(self, manager):
        """Test that model registry loads correctly."""
        assert "models" in manager.model_registry
        assert "providers" in manager.model_registry
        assert len(manager.model_registry["models"]) > 0
    
    def test_validate_model_valid(self, manager):
        """Test validation of valid models."""
        # Test some known models from our registry
        assert manager.validate_model("openai/gpt-4o")
        assert manager.validate_model("anthropic/claude-3.5-sonnet")
        assert manager.validate_model("mistral-7b-instruct")
    
    def test_validate_model_invalid(self, manager):
        """Test validation of invalid models."""
        assert not manager.validate_model("invalid/model")
        assert not manager.validate_model("gemini-1.5-pro")  # Gemini should not be in OpenRouter registry
        assert not manager.validate_model("")
        assert not manager.validate_model(None)
    
    def test_get_model_config_valid(self, manager):
        """Test getting configuration for valid models."""
        config = manager.get_model_config("openai/gpt-4o")
        assert "display_name" in config
        assert "max_temperature" in config
        assert "provider" in config
        assert config["provider"] == "OpenAI"
    
    def test_get_model_config_invalid(self, manager):
        """Test getting configuration for invalid models raises error."""
        with pytest.raises(ValueError, match="Unsupported OpenRouter model"):
            manager.get_model_config("invalid/model")
    
    def test_get_max_temperature_for_model(self, manager):
        """Test temperature limits for different models."""
        # OpenAI models should support up to 2.0
        assert manager.get_max_temperature_for_model("openai/gpt-4o") == 2.0
        
        # Anthropic models should support up to 1.0
        assert manager.get_max_temperature_for_model("anthropic/claude-3.5-sonnet") == 1.0
        
        # Mistral models should support up to 1.0
        assert manager.get_max_temperature_for_model("mistral-small-latest") == 1.0
        
        # Unknown models should default to 2.0
        assert manager.get_max_temperature_for_model("unknown/model") == 2.0
    
    def test_constrain_temperature(self, manager):
        """Test temperature constraint functionality."""
        # Test normal range
        assert manager.constrain_temperature(0.5, "openai/gpt-4o") == 0.5
        
        # Test upper bound constraint for Anthropic
        assert manager.constrain_temperature(2.0, "anthropic/claude-3.5-sonnet") == 1.0
        
        # Test lower bound constraint
        assert manager.constrain_temperature(-0.5, "openai/gpt-4o") == 0.0
        
        # Test upper bound constraint for OpenAI
        assert manager.constrain_temperature(3.0, "openai/gpt-4o") == 2.0
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
    def test_create_llm_valid_model(self, manager):
        """Test LLM creation for valid models."""
        llm = manager.create_llm("openai/gpt-4o", 0.7)
        assert llm.model == "openai/gpt-4o"
        assert llm.temperature == 0.7
        assert llm.openai_api_key == "test_key"
        assert llm.base_url == "https://openrouter.ai/api/v1"
    
    def test_create_llm_no_api_key(self, manager):
        """Test LLM creation fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is required"):
                manager.create_llm("openai/gpt-4o", 0.7)
    
    def test_create_llm_invalid_model(self, manager):
        """Test LLM creation fails for invalid models."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="Unsupported OpenRouter model"):
                manager.create_llm("invalid/model", 0.7)
    
    def test_get_supported_models(self, manager):
        """Test getting list of supported models."""
        models = manager.get_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "openai/gpt-4o" in models
        assert "anthropic/claude-3.5-sonnet" in models
    
    def test_get_models_by_provider(self, manager):
        """Test filtering models by provider."""
        openai_models = manager.get_models_by_provider("OpenAI")
        assert all("openai/" in model for model in openai_models)
        
        anthropic_models = manager.get_models_by_provider("Anthropic")
        assert all("anthropic/" in model for model in anthropic_models)
    
    def test_get_models_by_pricing_tier(self, manager):
        """Test filtering models by pricing tier."""
        free_models = manager.get_models_by_pricing_tier("free")
        premium_models = manager.get_models_by_pricing_tier("premium")
        
        assert len(free_models) > 0
        assert len(premium_models) > 0
        assert "google/gemma-2-9b-it:free" in free_models
        assert "openai/gpt-4o" in premium_models
    
    def test_handle_api_error(self, manager):
        """Test API error handling."""
        # Test quota error
        quota_error = Exception("quota exceeded")
        msg = manager.handle_api_error(quota_error, "openai/gpt-4o")
        assert "quota" in msg.lower()
        
        # Test rate limit error
        rate_error = Exception("rate limit exceeded")
        msg = manager.handle_api_error(rate_error, "openai/gpt-4o")
        assert "rate limit" in msg.lower()
        
        # Test generic error
        generic_error = Exception("unknown error")
        msg = manager.handle_api_error(generic_error, "openai/gpt-4o")
        assert "OpenRouter API error" in msg


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_is_openrouter_model(self):
        """Test OpenRouter model detection."""
        # Should detect valid OpenRouter models
        assert is_openrouter_model("openai/gpt-4o")
        assert is_openrouter_model("anthropic/claude-3.5-sonnet")
        
        # Should not detect Mistral models
        assert not is_openrouter_model("mistral-small-latest")
        
        # Should not detect Gemini models
        assert not is_openrouter_model("gemini-2.5-flash")
        assert not is_openrouter_model("gemini-1.5-pro")
        
        # Should not detect invalid models
        assert not is_openrouter_model("invalid/model")
        assert not is_openrouter_model("")
        assert not is_openrouter_model(None)
    
    def test_constrain_temperature_for_model_gemini(self):
        """Test temperature constraints for Gemini models."""
        # Gemini models should be constrained to 1.0
        assert constrain_temperature_for_model(2.0, "gemini-2.5-flash") == 1.0
        assert constrain_temperature_for_model(0.5, "gemini-2.5-flash") == 0.5
        assert constrain_temperature_for_model(-0.1, "gemini-2.5-flash") == 0.0
    
    def test_constrain_temperature_for_model_openrouter(self):
        """Test temperature constraints for OpenRouter models."""
        # OpenAI models should support up to 2.0
        assert constrain_temperature_for_model(1.5, "openai/gpt-4o") == 1.5
        assert constrain_temperature_for_model(3.0, "openai/gpt-4o") == 2.0
        
        # Anthropic models should be constrained to 1.0
        assert constrain_temperature_for_model(1.5, "anthropic/claude-3.5-sonnet") == 1.0
        assert constrain_temperature_for_model(0.8, "anthropic/claude-3.5-sonnet") == 0.8
    
    def test_constrain_temperature_for_model_unknown(self):
        """Test temperature constraints for unknown models."""
        # Unknown models should default to 2.0 max
        assert constrain_temperature_for_model(3.0, "unknown/model") == 2.0
        assert constrain_temperature_for_model(1.5, "unknown/model") == 1.5


class TestAgentIntegration:
    """Test cases for agent integration with OpenRouter."""
    
    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test_openrouter_key",
        "GOOGLE_API_KEY": "test_google_key", 
        "TAVILY_API_KEY": "test_tavily_key",
        "MISTRAL_API_KEY": "test_mistral_key"
    })
    def test_create_agent_with_openrouter_model(self):
        """Test agent creation with OpenRouter models."""
        # Test with a valid OpenRouter model
        agent = create_agent(
            model="openai/gpt-4o",
            temperature=0.7,
            verbosity=3
        )
        
        assert agent is not None
        assert hasattr(agent, 'system_prompt')
    
    @patch.dict(os.environ, {
        "GOOGLE_API_KEY": "test_google_key",
        "TAVILY_API_KEY": "test_tavily_key"
    })
    def test_create_agent_missing_openrouter_key(self):
        """Test agent creation fails without OpenRouter API key."""
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            create_agent(
                model="openai/gpt-4o",
                temperature=0.7,
                verbosity=3
            )
    
    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test_openrouter_key",
        "GOOGLE_API_KEY": "test_google_key",
        "TAVILY_API_KEY": "test_tavily_key"
    })
    def test_create_agent_invalid_model(self):
        """Test agent creation with invalid OpenRouter model."""
        with pytest.raises(ValueError):
            create_agent(
                model="invalid/model",
                temperature=0.7,
                verbosity=3
            )


if __name__ == "__main__":
    # Run basic validation tests
    print("Running OpenRouter integration validation...")
    
    # Test model manager initialization
    try:
        manager = OpenRouterModelManager()
        print("✓ OpenRouter model manager initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize model manager: {e}")
        exit(1)
    
    # Test model validation
    test_models = [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet", 
        "mistral-7b-instruct",
        "invalid/model"
    ]
    
    for model in test_models:
        is_valid = manager.validate_model(model)
        expected = model != "invalid/model"
        if is_valid == expected:
            print(f"✓ Model validation for '{model}': {is_valid}")
        else:
            print(f"✗ Model validation for '{model}': expected {expected}, got {is_valid}")
    
    # Test temperature constraints
    test_cases = [
        ("openai/gpt-4o", 1.5, 1.5),
        ("openai/gpt-4o", 3.0, 2.0),
        ("anthropic/claude-3.5-sonnet", 1.5, 1.0),
        ("gemini-2.5-flash", 2.0, 1.0),
    ]
    
    for model, temp_in, temp_expected in test_cases:
        temp_out = constrain_temperature_for_model(temp_in, model)
        if temp_out == temp_expected:
            print(f"✓ Temperature constraint for '{model}': {temp_in} -> {temp_out}")
        else:
            print(f"✗ Temperature constraint for '{model}': {temp_in} -> {temp_out}, expected {temp_expected}")
    
    print("OpenRouter integration validation completed!")