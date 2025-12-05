# app/core/openrouter_client.py
from typing import Dict, Any, List, Optional, Tuple
from openrouter import OpenRouter
from openrouter.components import Message
import base64
import json
import os
import logging

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    OpenRouter SDK client wrapper for AI operations.
    Provides unified interface for chart analysis and model management.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize OpenRouter client with API key.
        
        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        self.client = OpenRouter(api_key=api_key)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Fetch available models from OpenRouter.
        
        Returns:
            List of model dictionaries with id, name, and metadata
        """
        try:
            # Use the models endpoint to get available models
            # Note: OpenRouter SDK doesn't have a direct models.list() method
            # We'll use the HTTP endpoint directly
            import httpx
            
            response = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract model list from response
            models = data.get("data", [])
            
            # Filter for vision-capable models and return simplified list
            return [
                {
                    "id": model.get("id"),
                    "name": model.get("name", model.get("id")),
                    "context_length": model.get("context_length"),
                    "pricing": model.get("pricing", {}),
                    "architecture": model.get("architecture", {}),
                }
                for model in models
            ]
        except Exception as e:
            print(f"[OpenRouter] Failed to fetch models: {e}")
            # Return a fallback list of popular vision models
            return [
                {"id": "openai/gpt-4-vision-preview", "name": "GPT-4 Vision Preview"},
                {"id": "openai/gpt-4o", "name": "GPT-4o"},
                {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus"},
                {"id": "anthropic/claude-3-sonnet", "name": "Claude 3 Sonnet"},
                {"id": "google/gemini-pro-vision", "name": "Gemini Pro Vision"},
            ]
    
    def get_credits(self) -> Dict[str, Any]:
        """
        Fetch credit balance and usage information for this API key.
        
        Returns:
            Dictionary with credit balance, rate limits, and usage stats
        """
        try:
            import httpx
            
            response = httpx.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "credits": data.get("data", {}).get("limit_remaining", 0),
                "limit": data.get("data", {}).get("limit", 0),
                "usage": data.get("data", {}).get("usage", 0),
                "is_free_tier": data.get("data", {}).get("is_free_tier", True),
                "rate_limit": data.get("data", {}).get("rate_limit", {}),
            }
        except Exception as e:
            logger.error(f"[OpenRouter] Failed to fetch credits: {e}")
            return {
                "credits": 0,
                "limit": 0,
                "usage": 0,
                "is_free_tier": True,
                "rate_limit": {},
                "error": str(e)
            }
    
    def quick_inspect(self, image_bytes: bytes, ocr_text: str = "", model: Optional[str] = None) -> Dict[str, Any]:
        """
        Fast vision analysis for quick chart inspection.
        
        Args:
            image_bytes: Image data as bytes
            ocr_text: Optional OCR text hint
            model: Model ID to use (defaults to gpt-4o)
            
        Returns:
            Dictionary with symbol_guess, asset_type_guess, source_guess, is_relevant, confidence, excerpt
        """
        if not model:
            model = "openai/gpt-4o"
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        # Build prompt for quick inspection
        prompt = """You are a vision model that inspects a single image and returns ONLY a compact JSON summary.

IMPORTANT:
- Detect the primary instrument symbol EXACTLY AS PRINTED on the chart (preserve separators like "/", "-", ":" and original case).
- Do NOT guess a normalized symbol unless it is visibly printed; prefer on-image text over watermarks or side legends.
- Classify the asset type as one of: forex | crypto | index | stock | commodity | other.
- Detect the platform/source if a credible watermark/logo is visible (e.g., TradingView, Binance, Coinbase, Bybit, KuCoin, Investing.com, Yahoo Finance, Bloomberg, MetaTrader).
- Set is_relevant=true ONLY if this looks like a trading/chart/graph image (axes, candles/lines, OHLC, indicators, price/time scales, etc.)
- confidence: integer 0..100 for your extraction quality.
- excerpt: a short, readable single-line phrase (<= 120 chars), e.g. "BTC/USDT H1 crypto via TradingView".
- Return a single JSON object with keys exactly: symbol_guess, asset_type_guess, source_guess, is_relevant, confidence, excerpt.
- No code fences. No extra text."""
        
        if ocr_text:
            prompt += f"\n\nOCR_HINT (optional, do not overfit; prefer on-chart text):\n{ocr_text[:4000]}"
        
        try:
            # Send request using OpenRouter SDK
            response = self.client.chat.send(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=512,
                stream=False
            )
            
            # Extract response text
            content = response.choices[0].message.content if response.choices else ""
            
            # Parse JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', content)
                if json_match:
                    return json.loads(json_match.group(0))
                raise
                
        except Exception as e:
            print(f"[OpenRouter] Quick inspect failed: {e}")
            raise Exception(f"Quick inspect failed: {str(e)}")
    
    def generate_content(
        self,
        image_bytes: bytes,
        prompt: str,
        system_instruction: str = "",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Generate AI analysis for chart with vision + text.
        
        Args:
            image_bytes: Image data as bytes
            prompt: User prompt/instructions
            system_instruction: System instruction (prepended to prompt)
            model: Model ID to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON response from the model
        """
        if not model:
            model = "openai/gpt-4o"
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        # Combine system instruction and prompt
        full_prompt = system_instruction + "\n\n" + prompt if system_instruction else prompt
        
        try:
            # Send request using OpenRouter SDK
            response = self.client.chat.send(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": full_prompt}
                        ]
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                response_format={"type": "json_object"}
            )
            
            # Extract response text
            content = response.choices[0].message.content if response.choices else ""
            
            # Parse JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                # Find largest JSON object
                json_objects = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
                if json_objects:
                    # Sort by length and try longest first
                    json_objects.sort(key=len, reverse=True)
                    for obj in json_objects:
                        try:
                            return json.loads(obj)
                        except:
                            continue
                raise
                
        except Exception as e:
            print(f"[OpenRouter] Generate content failed: {e}")
            raise Exception(f"AI analysis failed: {str(e)}")


def get_openrouter_client(api_key: str) -> OpenRouterClient:
    """
    Factory function to create OpenRouter client.
    
    Args:
        api_key: OpenRouter API key
        
    Returns:
        Configured OpenRouterClient instance
    """
    if not api_key:
        raise ValueError("OpenRouter API key is required")
    
    return OpenRouterClient(api_key=api_key)


def get_all_credits(api_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch credits for multiple API keys.
    
    Args:
        api_keys: List of OpenRouter API keys
        
    Returns:
        List of credit info dictionaries with index
    """
    results = []
    for index, key in enumerate(api_keys):
        try:
            client = OpenRouterClient(api_key=key)
            credits = client.get_credits()
            credits["index"] = index
            results.append(credits)
        except Exception as e:
            logger.error(f"[OpenRouter] Failed to get credits for key {index}: {e}")
            results.append({
                "index": index,
                "credits": 0,
                "error": str(e)
            })
    
    return results


import re

def analyze_with_retry(
    image_bytes: bytes,
    prompt: str,
    api_keys: List[str],
    model: Optional[str] = None,
    system_instruction: str = "",
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> Tuple[Dict[str, Any], int]:
    """
    Attempt AI analysis with automatic retry on credit/rate limit failures.
    Also attempts to reduce max_tokens if the error indicates insufficient credits for the requested amount.
    
    Args:
        image_bytes: Image data as bytes
        prompt: User prompt/instructions
        api_keys: List of API keys to try in order
        model: Model ID to use
        system_instruction: System instruction
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (result dict, successful_key_index)
        
    Raises:
        Exception: If all keys fail or non-retryable error occurs
    """
    if not api_keys:
        raise ValueError("No API keys provided")
    
    last_error = None
    
    for index, api_key in enumerate(api_keys):
        try:
            logger.info(f"[OpenRouter] Attempting analysis with key {index + 1}/{len(api_keys)}")
            
            client = OpenRouterClient(api_key=api_key)
            result = client.generate_content(
                image_bytes=image_bytes,
                prompt=prompt,
                system_instruction=system_instruction,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info(f"[OpenRouter] ✓ Analysis successful with key {index + 1}")
            return (result, index)
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if this is a retryable error (402 = insufficient credits, 429 = rate limit)
            is_insufficient_credits = "402" in error_str or "insufficient" in error_str or "credits" in error_str
            is_rate_limit = "429" in error_str or "rate limit" in error_str
            
            if is_insufficient_credits:
                # Try to parse "can only afford X" from error message
                # Example: "You requested up to 4096 tokens, but can only afford 2728"
                affordable_match = re.search(r'can only afford (\d+)', error_str)
                
                if affordable_match:
                    affordable_tokens = int(affordable_match.group(1))
                    reduced_tokens = max(500, affordable_tokens - 50) # Leave a small buffer, min 500
                    
                    if reduced_tokens < max_tokens:
                        logger.warning(f"[OpenRouter] Key {index + 1} has limited credits. Retrying with {reduced_tokens} tokens...")
                        try:
                            client = OpenRouterClient(api_key=api_key)
                            result = client.generate_content(
                                image_bytes=image_bytes,
                                prompt=prompt,
                                system_instruction=system_instruction,
                                model=model,
                                temperature=temperature,
                                max_tokens=reduced_tokens
                            )
                            logger.info(f"[OpenRouter] ✓ Analysis successful with key {index + 1} (reduced tokens)")
                            return (result, index)
                        except Exception as retry_e:
                            logger.warning(f"[OpenRouter] Retry with reduced tokens failed: {retry_e}")
                            # Fall through to try next key
                
                # If parsing failed or retry failed, try a generic reduction if original was high
                elif max_tokens > 2000:
                    reduced_tokens = 2000
                    logger.warning(f"[OpenRouter] Key {index + 1} insufficient credits. Retrying with {reduced_tokens} tokens...")
                    try:
                        client = OpenRouterClient(api_key=api_key)
                        result = client.generate_content(
                            image_bytes=image_bytes,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            model=model,
                            temperature=temperature,
                            max_tokens=reduced_tokens
                        )
                        logger.info(f"[OpenRouter] ✓ Analysis successful with key {index + 1} (reduced tokens)")
                        return (result, index)
                    except Exception as retry_e:
                        logger.warning(f"[OpenRouter] Retry with reduced tokens failed: {retry_e}")
                
                logger.warning(f"[OpenRouter] Key {index + 1} exhausted, trying next key...")
                last_error = e
                continue  # Try next key
            
            elif is_rate_limit:
                logger.warning(f"[OpenRouter] Key {index + 1} hit rate limit, trying next key...")
                last_error = e
                continue  # Try next key
            
            else:
                # Non-retryable error (e.g., invalid request, model error)
                logger.error(f"[OpenRouter] Non-retryable error with key {index + 1}: {e}")
                raise  # Don't try other keys for non-retryable errors
    
    # All keys exhausted
    logger.error(f"[OpenRouter] All {len(api_keys)} API keys exhausted")
    if last_error:
        raise Exception(f"All API keys failed. Last error: {str(last_error)}")
    else:
        raise Exception("All API keys failed with unknown errors")


def quick_inspect_with_retry(
    image_bytes: bytes,
    api_keys: List[str],
    ocr_text: str = "",
    model: Optional[str] = None
) -> Tuple[Dict[str, Any], int]:
    """
    Quick inspect with automatic retry on credit/rate limit failures.
    
    Args:
        image_bytes: Image data as bytes
        api_keys: List of API keys to try in order
        ocr_text: Optional OCR text hint
        model: Model ID to use
        
    Returns:
        Tuple of (result dict, successful_key_index)
        
    Raises:
        Exception: If all keys fail or non-retryable error occurs
    """
    if not api_keys:
        raise ValueError("No API keys provided")
    
    last_error = None
    
    for index, api_key in enumerate(api_keys):
        try:
            logger.info(f"[OpenRouter] Quick inspect with key {index + 1}/{len(api_keys)}")
            
            client = OpenRouterClient(api_key=api_key)
            result = client.quick_inspect(
                image_bytes=image_bytes,
                ocr_text=ocr_text,
                model=model
            )
            
            logger.info(f"[OpenRouter] ✓ Quick inspect successful with key {index + 1}")
            return (result, index)
            
        except Exception as e:
            error_str = str(e).lower()
            
            is_insufficient_credits = "402" in error_str or "insufficient" in error_str or "credits" in error_str
            is_rate_limit = "429" in error_str or "rate limit" in error_str
            
            if is_insufficient_credits or is_rate_limit:
                logger.warning(f"[OpenRouter] Key {index + 1} failed (credits/rate limit), trying next...")
                last_error = e
                continue
            else:
                logger.error(f"[OpenRouter] Non-retryable error: {e}")
                raise
    
    logger.error(f"[OpenRouter] All {len(api_keys)} API keys exhausted for quick inspect")
    if last_error:
        raise Exception(f"All API keys failed. Last error: {str(last_error)}")
    else:
        raise Exception("All API keys failed with unknown errors")
