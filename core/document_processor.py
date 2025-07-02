import json
import re
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
from core.api_config import APIConfiguration

logger.add("logs/document_processor.log", rotation="100 MB", level="DEBUG", backtrace=True, diagnose=True)

class MetadataExtraction(BaseModel):
    contract_name: str
    parties: List[str]
    date: Optional[str] = None

class RelevanceAssessment(BaseModel):
    when_to_include: List[str]
    when_to_exclude: List[str]
    industry_considerations: List[str]
    risk_implications: List[str]
    compliance_requirements: List[str]
    best_practices: List[str]

class ClauseData(BaseModel):
    clause_type: str
    clause_text: str
    position_context: str
    clause_purpose: str
    relevance_assessment: RelevanceAssessment

class ClauseExtraction(BaseModel):
    clauses: List[ClauseData]

class DocumentProcessor:
    """Simplified document processor - AI operations only"""
    
    def __init__(self, api_config: APIConfiguration):
        self.api_config = api_config
        self.logger = logger
    
    def extract_metadata_from_text(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract contract metadata using AI"""
        self.logger.info(f"Extracting metadata for {filename}")
        
        if not self.api_config.is_configured():
            self.logger.warning("API not configured, returning default metadata")
            return {
                "contract_name": filename,
                "parties": ["Party A", "Party B"],
                "date": None,
                "filename": filename
            }
        
        try:
            prompt = f"""You are an expert legal document analyzer. Read the legal contract text and extract metadata. Return ONLY a JSON object with these exact fields:
{{
    "contract_name": "name or title of the contract",
    "parties": ["Party 1", "Party 2"],
    "date": "contract date (The date from when the contract is effective) in YYYY-MM-DD format if found, otherwise null"
}}

Do not include any other text or explanation.

Contract text (first 2000 characters): {text[:2000]}"""

            self.logger.debug("Sending request to AI for metadata extraction")
            response = self.api_config.generate_text(
                prompt=prompt,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            if response:
                self.logger.debug(f"Received metadata response: {response}")
                
                try:
                    metadata = json.loads(response)
                    
                    # Validate required fields
                    required_fields = ["contract_name", "parties", "date"]
                    for field in required_fields:
                        if field not in metadata:
                            self.logger.warning(f"Missing required field in metadata: {field}")
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Ensure parties is a list
                    if not isinstance(metadata["parties"], list):
                        self.logger.warning("Parties field is not a list, converting to list")
                        metadata["parties"] = [metadata["parties"]] if metadata["parties"] else []
                    
                    self.logger.info(f"Successfully extracted metadata: {metadata}")
                    metadata["filename"] = filename
                    return metadata
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse metadata JSON: {e}")
                    # Try to extract JSON from the response if it's wrapped in other text
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        try:
                            metadata = json.loads(json_match.group())
                            self.logger.info("Successfully extracted JSON using regex fallback")
                            metadata["filename"] = filename
                            return metadata
                        except json.JSONDecodeError:
                            self.logger.error("Failed to parse JSON even with regex fallback")
                            pass
                            
                except ValueError as ve:
                    self.logger.error(f"Metadata validation error: {ve}", exc_info=True)
                    
                # If all parsing attempts fail, return default metadata
                self.logger.warning("All metadata parsing attempts failed, returning default metadata")
                return {
                    "contract_name": filename,
                    "parties": [],
                    "date": None,
                    "filename": filename
                }
            else:
                return {
                    "contract_name": filename,
                    "parties": [],
                    "date": None,
                    "filename": filename
                }
                
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {str(e)}", exc_info=True)
            return {
                "contract_name": filename,
                "parties": [],
                "date": None,
                "filename": filename
            }
    
    def extract_clauses_from_text(self, contract_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract comprehensive clauses from contract with detailed metadata"""
        self.logger.info(f"Extracting clauses from contract: {metadata.get('contract_name', 'Unknown')}")
        
        if not self.api_config.is_configured():
            self.logger.warning("API not configured, returning empty clause list")
            return []
        
        prompt = f"""You are an experienced solicitor with decades of experience and are the senior partner of an international law firm.
You will be given a contract document text. Your task is to extract all distinct legal clauses from the contract, categorize them, and assess their relevance.

For each clause, identify:
1. Clause type (e.g., definitions, payment, termination, liability, intellectual_property, confidentiality, dispute_resolution, governing_law, force_majeure, etc.)
2. The complete clause text (preserve exact wording)
3. Its position/context in the contract
4. Clause purpose/function
5. Relevance assessment:
   - When should this clause be included?
   - When should it be excluded?
   - Industry-specific considerations
   - Risk implications
   - Compliance requirements
   - Best practices for implementation

IMPORTANT: Inside any field that contains text from the contract (like "clause_text"), you MUST escape any double quotes (") with a backslash (\\"). For example, if the text is 'The term "Agreement"...', it must be represented in the JSON as '"clause_text": "The term \\"Agreement\\"..."'. This is critical for the JSON to be valid. Also, ensure the JSON is well-formed and valid. Double check that the JSON is valid and contains all closing brackets and commas where needed.

Return as a JSON object with "clauses" array. Below is an example of the expected JSON structure:
{{
  "clauses": [
    {{
      "clause_type": "payment",
      "clause_text": "complete exact clause text...",
      "position_context": "Section 3.1 - Payment Terms",
      "clause_purpose": "Establishes payment obligations and timing",
      "relevance_assessment": {{
        "when_to_include": ["list of scenarios"],
        "when_to_exclude": ["list of scenarios"],
        "industry_considerations": ["specific to industry"],
        "risk_implications": ["key risks"],
        "compliance_requirements": ["relevant regulations"],
        "best_practices": ["implementation guidance"]
      }}
    }}
  ]
}}

Extract ALL distinct legal clauses from the contract with comprehensive categorization and relevance assessment.
Contract: 
{contract_text}"""
        
        try:
            self.logger.debug("Sending request to AI for clause extraction")
            
            try:
                # Use schema-based generation for better structured output
                response = self.api_config.generate_with_schema(
                    prompt=prompt,
                    schema_class=ClauseExtraction,
                    temperature=0.1
                )
            except Exception as gen_err:
                # Handle connection issues and fallback to basic generation
                self.logger.error(f"Schema-based generation failed: {gen_err}. Falling back to simple generation")
                try:
                    response = self.api_config.generate_text(
                        prompt=prompt,
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                except Exception as fallback_err:
                    self.logger.error(f"Fallback generation also failed: {fallback_err}")
                    return []
            
            self.logger.debug(f"Received clauses response: {response}")
            
            # Parse the JSON response
            try:
                clauses_json = json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse clauses JSON: {e}. For file: {metadata.get('contract_name', 'Unknown')}", exc_info=True)
                
                # Retry with simpler prompt
                retry_prompt = f"""This JSON response is not valid. 
The error is: {e} 
Fix the error and please ensure the JSON is well-formed and valid. Double check that the JSON is valid and contains all closing brackets and commas where needed. 
Rewrite the response with valid JSON format:
{response}"""

                retry_response = self.api_config.generate_text(
                    prompt=retry_prompt,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

                if retry_response:
                    clauses_json = json.loads(retry_response)
                    self.logger.debug(f"Received clauses response after retry: {retry_response}")
                else:
                    self.logger.error("Failed to get valid response after retry")
                    return []
            
            clauses = clauses_json.get("clauses", [])
            
            # Validate that we got a list
            if not isinstance(clauses, list):
                self.logger.warning("Clauses is not a list, attempting to convert")
                raise ValueError("Clauses is not a list")
            
            # Validate each clause has required fields
            required_fields = ["clause_type", "clause_text", "position_context"]
            for clause in clauses:
                for field in required_fields:
                    if field not in clause:
                        self.logger.warning(f"Missing required field in clause: {field}")
                        clause[field] = "Unknown"  # Provide default instead of failing
            
            # Add metadata to each clause
            for clause in clauses:
                clause.update({
                    "source_contract": metadata.get("contract_name", "Unknown"),
                    "contract_filename": metadata.get("filename", "Unknown"),
                    "contract_parties": metadata.get("parties", []),
                    "contract_date": metadata.get("date", None),
                    "contract_metadata": metadata
                })
            
            self.logger.info(f"Successfully extracted {len(clauses)} clauses")
            return clauses

        except Exception as e:
            self.logger.error(f"Clause extraction failed: {str(e)}", exc_info=True)
            return []