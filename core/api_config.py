import os
import logging
from google import genai
from google.genai import types
from typing import Optional, Dict, Any

class APIConfiguration:
    """Centralized API configuration management for Gemini Developer API"""
    
    def __init__(self):
        self.logger = logging.getLogger('legal_template_generator.api_config')
        
        # Load Gemini API key from environment
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', "AIzaSyDy9myuyslplkPf2pM19iV9ZBDClrA675w")
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Initialize the Gemini client
        try:
            self.client = genai.Client(api_key=self.gemini_api_key)
            self.logger.info("Gemini client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
        
        # Create aliases for compatibility with existing code
        self.google_client = self.client
        self.openai_client = self.client  # Using Google client for OpenAI-style calls
    
    def generate_text(
        self,
        prompt: str,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.1,
        thinking_budget: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text using Gemini with enhanced error handling"""
        try:
            if not self.is_configured():
                raise ValueError("Gemini API not configured")
            
            # Configure generation parameters
            config = types.GenerateContentConfig(
                temperature=temperature
            )
            
            # Add thinking budget if specified
            if thinking_budget is not None:
                config.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
            
            # Handle response format for JSON responses
            if response_format and response_format.get("type") == "json_object":
                config.response_mime_type = "application/json"
            
            # Generate content
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini")
            
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Gemini text generation failed: {str(e)}", exc_info=True)
            raise
    
    def generate_text_stream(
        self,
        prompt: str,
        model: str = "gemini-2.5-pro"
    ) -> None:
        """Streamed text generation (prints chunks as they arrive)."""
        try:
            stream = self.client.models.generate_content_stream(
                model=model,
                contents=[prompt]
            )
            
            for chunk in stream:
                print(chunk.text, end="")
                
        except Exception as e:
            self.logger.error(f"Streamed text generation failed: {e}")
            raise
    
    def create_chat(self, model: str = "gemini-2.5-pro"):
        """Start a new multi-turn chat session."""
        try:
            return self.client.chats.create(model=model)
        except Exception as e:
            self.logger.error(f"Chat creation failed: {e}")
            raise
    
    def chat_send(
        self,
        chat,
        message: str,
        stream: bool = False
    ) -> Optional[str]:
        """
        Send a message to an existing chat.
        If stream=True, prints chunks as they arrive; otherwise returns full text.
        """
        try:
            if stream:
                for chunk in chat.send_message_stream(message):
                    print(chunk.text, end="")
                return None
            else:
                resp = chat.send_message(message)
                return resp.text
                
        except Exception as e:
            self.logger.error(f"Chat message failed: {e}")
            raise
    
    def generate_with_schema(
        self,
        prompt: str,
        schema_class,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.1
    ) -> str:
        """Generate content with a specific Pydantic schema for structured output."""
        try:
            resp = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema_class,
                    "temperature": temperature
                }
            )
            
            return resp.text
            
        except Exception as e:
            self.logger.error(f"Schema-based generation failed: {e}")
            raise
    
    def is_configured(self) -> bool:
        """Check if the API is properly configured."""
        is_configured = self.client is not None and self.gemini_api_key is not None
        if not is_configured:
            self.logger.warning("❌ Gemini API not properly configured - client or API key missing")
            self.logger.debug(f"Client exists: {self.client is not None}, API key exists: {self.gemini_api_key is not None}")
        else:
            self.logger.debug("✅ Gemini API properly configured")
        return is_configured