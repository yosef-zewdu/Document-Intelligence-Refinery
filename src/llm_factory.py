import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_llm_config() -> Dict[str, Any]:
    """
    Returns LLM configuration from environment variables.
    
    Returns:
        dict with keys: provider, model, api_key, available, base_url
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    
    config = {
        "provider": provider,
        "model": None,
        "api_key": None,
        "available": False,
        "base_url": None,
    }
    
    if provider == "openrouter":
        config["model"] = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
        config["api_key"] = os.getenv("OPENROUTER_API_KEY")
        config["available"] = bool(config["api_key"])
        config["base_url"] = "https://openrouter.ai/api/v1"
    elif provider == "groq":
        config["model"] = os.getenv("GROQ_MODEL", "llama-3.2-90b-vision-preview")
        config["api_key"] = os.getenv("GROQ_API_KEY")
        config["available"] = bool(config["api_key"])
        config["base_url"] = "https://api.groq.com/openai/v1"
    elif provider == "ollama":
        config["model"] = os.getenv("OLLAMA_MODEL", "deepseek-coder")
        config["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        config["api_key"] = "ollama"  # Ollama doesn't need real API key
        config["available"] = True  # Assume available if selected
    elif provider == "huggingface":
        config["model"] = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        config["api_key"] = os.getenv("HF_TOKEN")
        config["available"] = bool(config["api_key"])
    elif provider == "gemini":
        config["model"] = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        config["api_key"] = os.getenv("GOOGLE_API_KEY")
        config["available"] = bool(config["api_key"])
    else:
        logger.warning(f"Unknown LLM provider: {provider}")
    
    return config


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
    """
    Returns a LangChain ChatModel instance based on the provider.
    Supported providers: 'gemini', 'openrouter', 'groq', 'ollama', 'huggingface'.
    
    Args:
        provider: LLM provider name (defaults to LLM_PROVIDER env var)
        model: Model name (defaults to provider-specific env var)
        api_key: API key (defaults to provider-specific env var)
    
    Returns:
        BaseChatModel instance
    
    Raises:
        ValueError: If provider is unsupported or API key is missing
    """
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    provider = provider or os.getenv("LLM_PROVIDER", "openrouter").lower()
    
    if provider == "openrouter":
        model = model or os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=key,
            model=model,
            temperature=0,
        )
    
    elif provider == "groq":
        model = model or os.getenv("GROQ_MODEL", "llama-3.2-90b-vision-preview")
        key = api_key or os.getenv("GROQ_API_KEY")
        
        if not key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        logger.info(f"Using Groq model: {model}")
        
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            openai_api_key=key,
            model=model,
            temperature=0,
        )
    
    elif provider == "ollama":
        model = model or os.getenv("OLLAMA_MODEL", "deepseek-coder")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        logger.info(f"Using Ollama model: {model} at {base_url}")
        
        return ChatOpenAI(
            base_url=f"{base_url}/v1",
            openai_api_key="ollama",  # Ollama doesn't need real API key
            model=model,
            temperature=0,
        )
    
    elif provider == "gemini":
        model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=0,
        )
    
    elif provider == "huggingface":
        # HuggingFace doesn't have a direct LangChain chat model
        # This is handled separately in the indexer for text generation
        raise ValueError(
            "HuggingFace provider should be used directly via InferenceClient, "
            "not through get_llm(). Use get_llm_config() instead."
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: openrouter, groq, ollama, gemini, huggingface")
