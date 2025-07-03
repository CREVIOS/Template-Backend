"""
AI/LLM API operation utilities for the Legal Template Generator.
Centralizes all AI API calls, retry logic, response handling, and error management.
"""

import json
import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from core.api_config import APIConfiguration

logger.add("logs/ai_api_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# API CONFIGURATION AND VALIDATION
# ============================================================================

async def validate_api_configuration(api_config: APIConfiguration) -> bool:
    """Validate API configuration before making calls"""
    try:
        if not api_config:
            logger.error("API configuration is None")
            return False
        
        if not api_config.is_configured():
            logger.error("API configuration is not properly set up")
            return False
        
        logger.debug("API configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating API configuration: {e}")
        return False

def get_api_rate_limits() -> Dict[str, Any]:
    """Get current API rate limits and usage statistics"""
    try:
        # This would integrate with actual API monitoring
        return {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "current_usage": {
                "minute": 0,
                "hour": 0,
                "day": 0
            },
            "reset_times": {
                "minute": datetime.utcnow().replace(second=0, microsecond=0),
                "hour": datetime.utcnow().replace(minute=0, second=0, microsecond=0),
                "day": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting API rate limits: {e}")
        return {}

# ============================================================================
# CORE API CALL FUNCTIONS WITH RETRY LOGIC
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception))
)
async def call_ai_api(
    api_config: APIConfiguration,
    prompt: str,
    temperature: float = 0.1,
    response_format: Optional[Dict] = None,
    schema_class: Optional[BaseModel] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 300
) -> str:
    """Make AI API call with comprehensive retry logic and error handling"""
    try:
        # Validate configuration
        if not await validate_api_configuration(api_config):
            raise ValueError("API configuration validation failed")
        
        # Log API call details
        logger.info(f"Making AI API call - prompt length: {len(prompt)}, temperature: {temperature}")
        
        start_time = time.time()
        
        # Choose appropriate API method based on parameters
        if schema_class:
            response = api_config.generate_with_schema(
                prompt=prompt,
                schema_class=schema_class,
                temperature=temperature
            )
        else:
            response = api_config.generate_text(
                prompt=prompt,
                temperature=temperature,
                response_format=response_format
            )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Validate response
        if not response or len(response.strip()) == 0:
            logger.warning("Received empty response from AI API")
            raise ValueError("Empty response from AI API")
        
        logger.info(f"AI API call successful - response time: {response_time:.2f}s, response length: {len(response)}")
        
        return response
        
    except Exception as e:
        logger.error(f"AI API call failed: {str(e)}", exc_info=True)
        raise

async def call_ai_api_with_fallback(
    primary_config: APIConfiguration,
    fallback_config: Optional[APIConfiguration],
    prompt: str,
    **kwargs
) -> str:
    """Make AI API call with fallback to secondary provider if primary fails"""
    try:
        # Try primary API first
        return await call_ai_api(primary_config, prompt, **kwargs)
        
    except Exception as primary_error:
        logger.warning(f"Primary AI API failed: {primary_error}")
        
        if fallback_config and await validate_api_configuration(fallback_config):
            try:
                logger.info("Attempting fallback API call")
                return await call_ai_api(fallback_config, prompt, **kwargs)
                
            except Exception as fallback_error:
                logger.error(f"Fallback AI API also failed: {fallback_error}")
                raise fallback_error
        else:
            logger.error("No valid fallback API configuration available")
            raise primary_error

# ============================================================================
# SPECIALIZED AI OPERATIONS
# ============================================================================

async def generate_text_with_validation(
    api_config: APIConfiguration,
    prompt: str,
    expected_format: str = "text",
    min_length: int = 10,
    max_length: int = 50000,
    **kwargs
) -> str:
    """Generate text with format validation and length checks"""
    try:
        response = await call_ai_api(api_config, prompt, **kwargs)
        
        # Validate response length
        if len(response) < min_length:
            logger.warning(f"Response too short: {len(response)} < {min_length}")
            raise ValueError(f"Response too short: expected at least {min_length} characters")
        
        if len(response) > max_length:
            logger.warning(f"Response too long: {len(response)} > {max_length}")
            response = response[:max_length]
            logger.info(f"Truncated response to {max_length} characters")
        
        # Format-specific validation
        if expected_format == "json":
            try:
                json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Response is not valid JSON: {e}")
                raise ValueError(f"Response is not valid JSON: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Text generation with validation failed: {e}")
        raise

async def generate_structured_response(
    api_config: APIConfiguration,
    prompt: str,
    schema_class: BaseModel,
    max_retries: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """Generate structured response using Pydantic schema with validation and repair"""
    try:
        for attempt in range(max_retries):
            response = None
            try:
                response = await call_ai_api(
                    api_config=api_config,
                    prompt=prompt,
                    schema_class=schema_class,
                    **kwargs
                )
                
                # Parse and validate response
                if isinstance(response, str):
                    structured_data = json.loads(response)
                else:
                    structured_data = response
                
                # Validate against schema
                validated_data = schema_class(**structured_data)
                
                logger.info(f"Successfully generated structured response on attempt {attempt + 1}")
                return validated_data.dict()
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Attempt {attempt + 1} failed with validation error: {e}")
                
                if attempt < max_retries - 1 and response:
                    # Try to repair the response
                    try:
                        repaired_response = await repair_json_response(api_config, response)
                        if repaired_response:
                            structured_data = json.loads(repaired_response)
                            validated_data = schema_class(**structured_data)
                            logger.info(f"Successfully repaired and validated response on attempt {attempt + 1}")
                            return validated_data.dict()
                    except Exception as repair_error:
                        logger.warning(f"JSON repair failed: {repair_error}")
                
                if attempt == max_retries - 1:
                    raise e
        
        raise ValueError(f"Failed to generate valid structured response after {max_retries} attempts")
        
    except Exception as e:
        logger.error(f"Structured response generation failed: {e}")
        raise

async def generate_json_response(
    api_config: APIConfiguration,
    prompt: str,
    expected_schema: Optional[Dict] = None,
    auto_repair: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Generate JSON response with automatic validation and repair"""
    try:
        response = await call_ai_api(
            api_config=api_config,
            prompt=prompt,
            response_format={"type": "json_object"},
            **kwargs
        )
        
        # Parse JSON
        try:
            json_data = json.loads(response)
            
            # Validate against expected schema if provided
            if expected_schema:
                validate_json_schema(json_data, expected_schema)
            
            return json_data
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            
            if auto_repair:
                logger.info("Attempting automatic JSON repair")
                repaired_response = await repair_json_response(api_config, response)
                
                if repaired_response:
                    json_data = json.loads(repaired_response)
                    
                    if expected_schema:
                        validate_json_schema(json_data, expected_schema)
                    
                    return json_data
            
            raise ValueError(f"Failed to parse JSON response: {e}")
        
    except Exception as e:
        logger.error(f"JSON response generation failed: {e}")
        raise

# ============================================================================
# RESPONSE PROCESSING AND REPAIR
# ============================================================================

async def repair_json_response(
    api_config: APIConfiguration,
    invalid_json: str,
    max_repair_attempts: int = 2
) -> Optional[str]:
    """Repair invalid JSON responses using AI"""
    try:
        # Identify the JSON error
        error_message = "Unknown JSON error"
        try:
            json.loads(invalid_json)
        except json.JSONDecodeError as e:
            error_message = str(e)
        
        repair_prompt = f"""You are a JSON repair expert. The following text is a JSON response that has validation errors or is malformed.

JSON Error: {error_message}

Your task is to fix the JSON and return a properly formatted valid JSON object. Common issues to fix:
1. Missing or extra commas
2. Unescaped quotes in string values
3. Missing closing brackets or braces
4. Trailing commas
5. Invalid escape sequences
6. Incorrect nesting of objects or arrays

CRITICAL REQUIREMENTS:
- Fix ALL JSON syntax errors
- Preserve all content and meaning from the original
- Ensure proper escaping of quotes in string values
- Return ONLY the fixed JSON - no explanations or comments
- The result must be valid, parseable JSON

Invalid JSON to repair:
{invalid_json}"""
        
        for attempt in range(max_repair_attempts):
            try:
                repaired_response = await call_ai_api(
                    api_config=api_config,
                    prompt=repair_prompt,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                # Validate the repaired JSON
                json.loads(repaired_response)
                logger.info(f"Successfully repaired JSON on attempt {attempt + 1}")
                return repaired_response
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON repair attempt {attempt + 1} failed: {e}")
                if attempt == max_repair_attempts - 1:
                    logger.error("All JSON repair attempts failed")
                    return None
        
        return None
        
    except Exception as e:
        logger.error(f"JSON repair process failed: {e}")
        return None

def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate JSON data against a simple schema"""
    try:
        # Basic schema validation
        for required_field in schema.get("required", []):
            if required_field not in data:
                raise ValueError(f"Missing required field: {required_field}")
        
        # Type validation
        for field, expected_type in schema.get("properties", {}).items():
            if field in data:
                if expected_type == "string" and not isinstance(data[field], str):
                    raise ValueError(f"Field {field} must be a string")
                elif expected_type == "array" and not isinstance(data[field], list):
                    raise ValueError(f"Field {field} must be an array")
                elif expected_type == "object" and not isinstance(data[field], dict):
                    raise ValueError(f"Field {field} must be an object")
        
        return True
        
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise

# ============================================================================
# BATCH PROCESSING AND ASYNC OPERATIONS
# ============================================================================

async def process_batch_requests(
    api_config: APIConfiguration,
    requests: List[Dict[str, Any]],
    batch_size: int = 5,
    delay_between_batches: float = 1.0
) -> List[Dict[str, Any]]:
    """Process multiple AI API requests in batches with rate limiting"""
    try:
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_results = []
            
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(requests) + batch_size - 1) // batch_size}")
            
            # Process batch concurrently (if API supports it) or sequentially
            for request in batch:
                try:
                    result = await call_ai_api(
                        api_config=api_config,
                        prompt=request.get("prompt", ""),
                        **request.get("params", {})
                    )
                    
                    batch_results.append({
                        "request_id": request.get("id"),
                        "success": True,
                        "response": result,
                        "error": None
                    })
                    
                except Exception as e:
                    logger.error(f"Batch request failed: {e}")
                    batch_results.append({
                        "request_id": request.get("id"),
                        "success": False,
                        "response": None,
                        "error": str(e)
                    })
            
            results.extend(batch_results)
            
            # Delay between batches to respect rate limits
            if i + batch_size < len(requests):
                await asyncio.sleep(delay_between_batches)
        
        logger.info(f"Completed batch processing: {len(results)} requests processed")
        return results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise

# ============================================================================
# CONTENT PROCESSING AND ANALYSIS
# ============================================================================

async def extract_metadata_from_text(
    api_config: APIConfiguration,
    text: str,
    metadata_fields: List[str],
    max_text_length: int = 8000
) -> Dict[str, Any]:
    """Extract structured metadata from text using AI"""
    try:
        # Truncate text if too long
        if len(text) > max_text_length:
            text = text[:max_text_length]
            logger.warning(f"Text truncated to {max_text_length} characters for metadata extraction")
        
        # Build metadata extraction prompt
        fields_list = ", ".join(metadata_fields)
        
        prompt = f"""Extract the following metadata from the provided text: {fields_list}

Requirements:
- Return the data in JSON format
- Use null for any information not found
- Ensure all requested fields are included in the response
- Be accurate and precise in extraction

Text to analyze:
{text}"""
        
        response = await generate_json_response(
            api_config=api_config,
            prompt=prompt,
            temperature=0.1
        )
        
        # Ensure all requested fields are present
        for field in metadata_fields:
            if field not in response:
                response[field] = None
        
        logger.info(f"Successfully extracted metadata: {list(response.keys())}")
        return response
        
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return {field: None for field in metadata_fields}

async def summarize_text(
    api_config: APIConfiguration,
    text: str,
    summary_length: str = "medium",
    focus_areas: Optional[List[str]] = None
) -> str:
    """Generate AI-powered text summary"""
    try:
        # Determine summary parameters
        length_instructions = {
            "short": "Generate a concise summary in 1-2 sentences.",
            "medium": "Generate a comprehensive summary in 1-2 paragraphs.",
            "long": "Generate a detailed summary with key points and important details."
        }
        
        length_instruction = length_instructions.get(summary_length, length_instructions["medium"])
        
        # Build prompt
        prompt = f"""Analyze and summarize the following text.

Requirements:
{length_instruction}
- Focus on the most important information
- Maintain accuracy and context"""
        
        if focus_areas:
            focus_list = ", ".join(focus_areas)
            prompt += f"\n- Pay special attention to: {focus_list}"
        
        prompt += f"\n\nText to summarize:\n{text}"
        
        summary = await call_ai_api(
            api_config=api_config,
            prompt=prompt,
            temperature=0.3
        )
        
        logger.info(f"Generated {summary_length} summary: {len(summary)} characters")
        return summary
        
    except Exception as e:
        logger.error(f"Text summarization failed: {e}")
        raise

# ============================================================================
# API MONITORING AND ANALYTICS
# ============================================================================

class APICallTracker:
    """Track API call statistics and performance"""
    
    def __init__(self):
        self.calls = []
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.errors = []
    
    def log_call(
        self,
        prompt_length: int,
        response_length: int,
        response_time: float,
        success: bool,
        error: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None
    ):
        """Log an API call for tracking"""
        call_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_length": prompt_length,
            "response_length": response_length,
            "response_time": response_time,
            "success": success,
            "error": error,
            "tokens_used": tokens_used or 0,
            "cost": cost or 0.0
        }
        
        self.calls.append(call_data)
        self.total_calls += 1
        
        if tokens_used:
            self.total_tokens += tokens_used
        if cost:
            self.total_cost += cost
        if error:
            self.errors.append(call_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive API call statistics"""
        if not self.calls:
            return {"total_calls": 0, "success_rate": 0, "average_response_time": 0}
        
        successful_calls = [call for call in self.calls if call["success"]]
        response_times = [call["response_time"] for call in successful_calls]
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": len(successful_calls),
            "failed_calls": len(self.errors),
            "success_rate": len(successful_calls) / self.total_calls * 100,
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "recent_errors": self.errors[-5:] if self.errors else []
        }

# Global tracker instance
api_tracker = APICallTracker()

def get_api_statistics() -> Dict[str, Any]:
    """Get current API usage statistics"""
    return api_tracker.get_statistics()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def prepare_prompt_for_api(
    prompt: str,
    max_length: int = 100000,
    preserve_structure: bool = True
) -> str:
    """Prepare and optimize prompt for API calls"""
    try:
        if len(prompt) <= max_length:
            return prompt
        
        logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {max_length}")
        
        if preserve_structure:
            # Try to preserve important sections
            lines = prompt.split('\n')
            truncated_lines = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 <= max_length:
                    truncated_lines.append(line)
                    current_length += len(line) + 1
                else:
                    break
            
            truncated_prompt = '\n'.join(truncated_lines)
            if len(truncated_prompt) < max_length - 100:
                truncated_prompt += "\n\n[Content truncated due to length limits]"
            
            return truncated_prompt
        else:
            # Simple truncation
            return prompt[:max_length] + "\n\n[Content truncated due to length limits]"
        
    except Exception as e:
        logger.error(f"Prompt preparation failed: {e}")
        return prompt[:max_length] if len(prompt) > max_length else prompt

def estimate_token_count(text: str, model: str = "gemini") -> int:
    """Estimate token count for text (rough approximation)"""
    try:
        # Rough estimation: ~4 characters per token for most models
        if model.lower() in ["gemini", "gpt"]:
            return len(text) // 4
        else:
            return len(text.split())  # Word count for other models
        
    except Exception as e:
        logger.error(f"Token estimation failed: {e}")
        return 0

async def chunk_text_for_processing(
    text: str,
    chunk_size: int = 8000,
    overlap: int = 200
) -> List[str]:
    """Split large text into chunks for processing"""
    try:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundaries
            if end < len(text):
                # Find the last space before the end
                space_index = text.rfind(' ', start, end)
                if space_index > start:
                    end = space_index
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if end < len(text) else end
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Text chunking failed: {e}")
        return [text] 