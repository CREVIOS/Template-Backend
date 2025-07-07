import json
import re
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
from core.api_config import APIConfiguration
from enum import Enum  # Added missing import
from functools import lru_cache
from supabase import create_client, Client

logger.add("logs/document_processor.log", rotation="100 MB", level="DEBUG", backtrace=True, diagnose=True)

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Get Supabase client for database operations"""
    from core.database import DatabaseSettings
    
    settings = DatabaseSettings()
    if not settings.supabase_url or not settings.supabase_service_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    
    return create_client(settings.supabase_url, settings.supabase_service_key)

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


class ClauseType(str, Enum):
    # Fundamental Clauses
    PAYMENT_TERMS = "Payment Terms Clause"
    SCOPE_OF_WORK = "Scope of Work (Statement of Work) Clause"
    TERM_TERMINATION = "Term & Termination Clause"
    PRICE_ADJUSTMENT = "Price Adjustment / Escalation Clause"
    
    # Protective Clauses
    INDEMNITY = "Indemnity Clause"
    LIMITATION_LIABILITY = "Limitation of Liability Clause"
    EXEMPTION_EXCLUSION = "Exemption / Exclusion Clause"
    LIQUIDATED_DAMAGES = "Liquidated Damages Clause"
    EXCULPATORY = "Exculpatory Clause"
    GROSS_UP = "Gross-Up Clause"
    RETENTION_TITLE = "Retention of Title (Romalpa) Clause"
    
    # Dispute Resolution Clauses
    ARBITRATION = "Arbitration Clause"
    DISPUTE_RESOLUTION = "Dispute Resolution / Escalation Clause"
    CHOICE_OF_LAW = "Choice of Law Clause"
    CONFESSION_JUDGMENT = "Confession of Judgment Clause"
    
    # Confidentiality & IP Clauses
    CONFIDENTIALITY = "Confidentiality / Non-Disclosure Clause"
    INTELLECTUAL_PROPERTY = "Intellectual Property Clause"
    NON_COMPETE = "Non-Compete Clause"
    NON_SOLICITATION = "Non-Solicitation Clause"
    
    # Operational (Boilerplate) Clauses
    ASSIGNMENT = "Assignment Clause"
    CHANGE_CONTROL = "Change Control / Changes Clause"
    AMENDMENT = "Amendment Clause"
    NOTICE = "Notice Clause"
    SEVERABILITY = "Severability Clause"
    SURVIVAL = "Survival Clause"
    ENTIRE_AGREEMENT = "Entire Agreement Clause"
    WAIVER = "Waiver Clause"
    INTERPRETATION = "Interpretation Clause"
    ELECTRONIC_SIGNATURES = "Electronic Signatures Clause"
    
class ClauseData(BaseModel):
    clause_type: ClauseType  # Changed from str to ClauseType enum
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
    
    def extract_metadata_from_text(self, text: str, filename: str, file_id: str = None, folder_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Extract contract metadata using AI and store in database if file_id is provided"""
        self.logger.info(f"Extracting metadata for {filename}")
        
        if not self.api_config.is_configured():
            self.logger.warning("API not configured, returning default metadata")
            default_metadata = {
                "contract_name": filename,
                "parties": ["Party A", "Party B"],
                "date": None,
                "summary": "",
                "filename": filename
            }
            
            # Store default metadata if file_id is provided
            if file_id and folder_id and user_id:
                self._store_metadata_in_db(default_metadata, file_id, folder_id, user_id)
                
            return default_metadata
        
        try:
            prompt = f"""You are an expert legal document analyzer. Read the legal contract text and extract metadata. Return ONLY a JSON object with these exact fields:
{{
    "contract_name": "name or title of the contract",
    "parties": ["Party 1", "Party 2"],
    "date": "contract date (The date from when the contract is effective) in YYYY-MM-DD format if found, otherwise null",
    "summary": "provide a concise summary of the contract in 5-7 sentences only"
}}

IMPORTANT: Return ONLY the JSON object with the exact fields specified above. Use strict JSON formatting with proper quotes and brackets. Do not include any additional text, explanations, or markdown formatting.

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
                    required_fields = ["contract_name", "parties", "date", "summary"]
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
                    
                    # Store metadata in database if file_id is provided
                    if file_id and folder_id and user_id:
                        self._store_metadata_in_db(metadata, file_id, folder_id, user_id)
                        
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
                            
                            # Store metadata in database if file_id is provided
                            if file_id and folder_id and user_id:
                                self._store_metadata_in_db(metadata, file_id, folder_id, user_id)
                                
                            return metadata
                        except json.JSONDecodeError:
                            self.logger.error("Failed to parse JSON even with regex fallback")
                            pass
                            
                except ValueError as ve:
                    self.logger.error(f"Metadata validation error: {ve}", exc_info=True)
                    
                # If all parsing attempts fail, return default metadata
                self.logger.warning("All metadata parsing attempts failed, returning default metadata")
                default_metadata = {
                    "contract_name": filename,
                    "parties": [],
                    "date": None,
                    "summary": "",
                    "filename": filename
                }
                
                # Store default metadata in database if file_id is provided
                if file_id and folder_id and user_id:
                    self._store_metadata_in_db(default_metadata, file_id, folder_id, user_id)
                    
                return default_metadata
            else:
                default_metadata = {
                    "contract_name": filename,
                    "parties": [],
                    "date": None,
                    "summary": "",
                    "filename": filename
                }
                
                # Store default metadata in database if file_id is provided
                if file_id and folder_id and user_id:
                    self._store_metadata_in_db(default_metadata, file_id, folder_id, user_id)
                    
                return default_metadata
                
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {str(e)}", exc_info=True)
            default_metadata = {
                "contract_name": filename,
                "parties": [],
                "date": None,
                "summary": "",
                "filename": filename
            }
            
            # Store default metadata in database if file_id is provided
            if file_id and folder_id and user_id:
                self._store_metadata_in_db(default_metadata, file_id, folder_id, user_id)
                
            return default_metadata
    
    def _store_metadata_in_db(self, metadata: Dict[str, Any], file_id: str, folder_id: str, user_id: str) -> bool:
        """Store metadata in the file_info table in Supabase"""
        try:
            self.logger.info(f"Storing metadata in database for file_id: {file_id}")
            
            # Create or get Supabase client
            supabase = get_supabase_client()
            
            # Prepare data for file_info table
            file_info_data = {
                "file_id": file_id,
                "user_id": user_id,
                "contract_infos": metadata
            }
            
            # Insert or update in file_info table
            result = supabase.table("file_info").upsert(file_info_data).execute()
            
            if result and hasattr(result, "data") and result.data:
                self.logger.info(f"Successfully stored metadata in database for file_id: {file_id}")
                return True
            else:
                self.logger.warning(f"No data returned when storing metadata for file_id: {file_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to store metadata in database: {str(e)}", exc_info=True)
            return False
    
    def extract_clauses_from_text(self, contract_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract comprehensive clauses from contract with detailed metadata"""
        self.logger.info(f"🔍 Starting clause extraction from contract: {metadata.get('contract_name', 'Unknown')}")
        
        if not self.api_config.is_configured():
            self.logger.error("❌ API not configured for clause extraction")
            raise ValueError("API not configured")
        
        if not contract_text or len(contract_text.strip()) == 0:
            self.logger.warning("❌ Empty contract text provided")
            return []
        
        prompt = f"""You are an experienced solicitor with decades of experience and are the senior partner of an international law firm.
        
    TASK:
    You will be given a contract document text. Your task is to extract all distinct legal clauses from the contract, categorize them using the predefined clause types, and assess their relevance.

    For each clause, identify:
    1. Clause type (MUST be one of the predefined types listed below)
    2. The complete clause text (preserve exact wording)
    3. Its position/context in the contract (e.g., "Section 3", "Article 5.2")
    4. Clause purpose/function
    5. Relevance assessment including:
    - When should this clause be included
    - When should it be excluded
    - Industry-specific considerations
    - Risk implications
    - Compliance requirements
    - Best practices for implementation

    PREDEFINED CLAUSE TYPES (you MUST use exactly these values):
    - Fundamental Clauses: "Payment Terms Clause", "Scope of Work (Statement of Work) Clause", "Term & Termination Clause", "Price Adjustment / Escalation Clause"
    - Protective Clauses: "Indemnity Clause", "Limitation of Liability Clause", "Exemption / Exclusion Clause", "Liquidated Damages Clause", "Exculpatory Clause", "Gross-Up Clause", "Retention of Title (Romalpa) Clause"
    - Dispute Resolution Clauses: "Arbitration Clause", "Dispute Resolution / Escalation Clause", "Choice of Law Clause", "Confession of Judgment Clause"
    - Confidentiality & IP Clauses: "Confidentiality / Non-Disclosure Clause", "Intellectual Property Clause", "Non-Compete Clause", "Non-Solicitation Clause"
    - Operational (Boilerplate) Clauses: "Assignment Clause", "Change Control / Changes Clause", "Amendment Clause", "Notice Clause", "Severability Clause", "Survival Clause", "Entire Agreement Clause", "Waiver Clause", "Interpretation Clause", "Electronic Signatures Clause"

    IMPORTANT INSTRUCTIONS:
    - Extract ALL distinct legal clauses from the contract
    - Each clause MUST be categorized using one of the predefined clause types listed above
    - Maintain the EXACT text of each clause
    - Ensure comprehensive relevance assessment for each clause
    - Do not omit any clauses, even if they seem standard
    - Ensure the output matches the required schema format exactly and is valid JSON

    IMPORTANT: Inside any field that contains text from the contract (like "clause_text"), you MUST escape any double quotes (") with a backslash (\\"). For example, if the text is 'The term "Agreement"...', it must be represented in the JSON as '"clause_text": "The term \\"Agreement\\"..."'. This is critical for the JSON to be valid.

    Return the result as a valid JSON object with a "clauses" array containing objects with the following structure:
    {{
    "clauses": [
        {{
        "clause_type": "Payment Terms Clause",
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

    Contract: 
    {contract_text}"""
        
        try:
            self.logger.debug(f"📤 Sending request to extract clauses from contract (content length: {len(contract_text)})")
            
            # Use schema-based generation for better structured output
            try:
                response = self.api_config.generate_with_schema(
                    prompt=prompt,
                    schema_class=ClauseExtraction,
                    temperature=0.1
                )
            except Exception as gen_err:
                # Handle connection issues and fallback to basic generation
                self.logger.error(f"❌ Schema-based generation failed: {gen_err}. Falling back to simple generation")
                try:
                    response = self.api_config.generate_text(
                        prompt=prompt,
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                except Exception as fallback_err:
                    self.logger.error(f"❌ Fallback generation also failed: {fallback_err}")
                    return []
            
            if not response or len(response.strip()) == 0:
                self.logger.warning("❌ Empty response from API when extracting clauses")
                return []
            
            self.logger.debug(f"📥 Received response from API (length: {len(response)})")
            
            # Parse the JSON response with comprehensive error handling
            try:
                # First attempt to parse the JSON response
                try:
                    extracted_data = json.loads(response)
                    clauses = extracted_data.get("clauses", [])
                    
                    if not isinstance(clauses, list):
                        self.logger.warning("⚠️  Extracted clauses is not a list, attempting JSON repair")
                        raise ValueError("Clauses is not a list")
                    
                except (json.JSONDecodeError, ValueError) as json_error:
                    # If there's a JSON parsing error, try to repair it with a second LLM call
                    self.logger.warning(f"⚠️  JSON validation failed: {str(json_error)}, attempting repair")
                    repaired_json = self._repair_json_response_clause_extraction(response)
                    
                    if not repaired_json:
                        self.logger.error("❌ Failed to repair JSON response")
                        return []
                    
                    extracted_data = json.loads(repaired_json)
                    clauses = extracted_data.get("clauses", [])
                    
                    if not isinstance(clauses, list):
                        self.logger.error("❌ Repaired JSON still does not contain a valid clauses list")
                        return []
                
                # Validate that we got a list
                if not isinstance(clauses, list):
                    self.logger.warning("⚠️  Clauses is not a list, attempting to convert")
                    raise ValueError("Clauses is not a list")
                
                # Validate each clause has required fields
                required_fields = ["clause_type", "clause_text", "position_context", "clause_purpose"]
                for clause in clauses:
                    for field in required_fields:
                        if field not in clause:
                            self.logger.warning(f"⚠️  Missing required field in clause: {field}")
                            clause[field] = "Unknown"  # Provide default instead of failing
                
                # Add metadata to each clause
                for clause in clauses:
                    clause.update({
                        "extracted_at": datetime.utcnow().isoformat(),
                        "source_contract": metadata.get("contract_name", "Unknown"),
                        "contract_filename": metadata.get("filename", "Unknown"),
                        "contract_parties": metadata.get("parties", []),
                        "contract_date": metadata.get("date", None),
                        "contract_metadata": metadata
                    })
                
                self.logger.info(f"✅ Successfully extracted {len(clauses)} clauses from contract")
                
                # Log the types of clauses extracted for debugging
                if clauses:
                    clause_types = {}
                    for clause in clauses:
                        clause_type = clause.get("clause_type", "unknown")
                        clause_types[clause_type] = clause_types.get(clause_type, 0) + 1
                    self.logger.info(f"📊 Clause types extracted: {clause_types}")
                
                return clauses
                
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Failed to parse JSON from API response: {str(e)}", exc_info=True)
                self.logger.debug(f"Raw API response: {response[:1000]}...")  # First 1000 chars for debugging
                return []

        except Exception as e:
            self.logger.error(f"❌ Clause extraction failed: {str(e)}", exc_info=True)
            return []


    def _repair_json_response_clause_extraction(self, response: str) -> str:
        """
        Repair malformed JSON response from clause extraction using LLM
        """
        self.logger.debug("🔧 Attempting to repair malformed JSON response")
        
        retry_prompt = f"""The following JSON response is malformed and needs to be fixed. 
    Please return a valid JSON object that follows the exact structure required for clause extraction.

    The JSON should have this structure:
    {{
    "clauses": [
        {{
        "clause_type": "one of the predefined clause types",
        "clause_text": "complete exact clause text",
        "position_context": "section/article reference",
        "clause_purpose": "description of clause purpose",
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

    IMPORTANT: 
    - Ensure all quotes are properly escaped
    - Ensure all brackets and braces are properly closed
    - Ensure all arrays and objects are properly formatted
    - Do not add any explanatory text, return ONLY the valid JSON

    Malformed JSON to repair:
    {response}"""

        try:
            repaired_response = self.api_config.generate_text(
                prompt=retry_prompt,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            if repaired_response:
                # Test if the repaired JSON is valid
                json.loads(repaired_response)
                self.logger.debug("✅ Successfully repaired JSON response")
                return repaired_response
            else:
                self.logger.error("❌ No response received for JSON repair")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Repaired JSON is still invalid: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"❌ JSON repair failed: {str(e)}")
            return None