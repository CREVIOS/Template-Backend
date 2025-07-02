import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from core.api_config import APIConfiguration
from pydantic import BaseModel

logger.add("logs/template_generator.log", rotation="100 MB", level="DEBUG", backtrace=True, diagnose=True)

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

class TemplateGenerator:
    """Generate legal templates using Gemini AI with enhanced features"""
    
    def __init__(self, api_config: APIConfiguration):
        self.api_config = api_config
        self.logger = logger
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_gemini(self, prompt: str, temperature: float = 0.1, response_format: Optional[Dict] = None, schema_class=None) -> str:
        """Make Gemini API call with retry logic"""
        try:
            if not self.api_config.is_configured():
                self.logger.error("Gemini API not configured")
                raise ValueError("Gemini API not configured")
            
            if schema_class:
                # Use schema-based generation for structured output
                return self.api_config.generate_with_schema(
                    prompt=prompt,
                    schema_class=schema_class,
                    temperature=temperature
                )
            else:
                # Use regular text generation
                return self.api_config.generate_text(
                    prompt=prompt,
                    temperature=temperature,
                    response_format=response_format
                )
            
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {str(e)}", exc_info=True)
            raise
    
    def generate_initial_template(self, contract_text: str) -> str:
        """Generate initial template from first contract with enhanced placeholders"""
        self.logger.info("Generating initial template")
        
        if not self.api_config.is_configured():
            self.logger.error("API not configured")
            raise ValueError("API not configured")
        
        prompt = f"""You are an experienced solicitor in Hong Kong. You have decades of experience and are the senior partner of an international law firm. Remember law is a practice of precision. Stick to the precedent's language and use of words as much as possible. If the clause is drafted long, then keep it long. If it's drafted short, keep it short. 

Attached is a legal document you recently drafted. You think it is well drafted and would like to turn it into a template so you can reuse it in the future easily. Please turn it into a template document.

IMPORTANT REQUIREMENTS:
1. Use square brackets for ALL placeholders that need customization, such as:
   - Party names: [Party Name], [Company Name], [Client Name]
   - Dates: [Effective Date], [Contract Date], [Termination Date], [Due Date]
   - Amounts: [Contract Amount], [Payment Amount], [Penalty Amount]
   - Addresses: [Company Address], [Service Address], [Registered Address]
   - Contact details: [Email Address], [Phone Number], [Contact Person]
   - Legal entities: [Legal Entity Type], [Registration Number], [Jurisdiction]
   - Terms: [Payment Terms], [Service Period], [Notice Period]
   - Any other variable information that would change between contracts

2. Be comprehensive - identify ALL possible fields that might vary between different uses of this template and clauses. 

3. Keep the legal language precise and professional.

4. CLAUSE LIST FORMATTING REQUIREMENTS FOR AUTOMATED PROCESSING AND WORD COMPATIBILITY:
   - ALL numbered items MUST use the following prefix formats - DO NOT use manual numbering like "1." or "1.1" directly:
     * Main clauses: "NUMBERED_LIST_ITEM: [content]" (replaces 1., 2., 3., etc.)
     * First-level numerical subclauses: "SUB_NUMBERED_ITEM: [content]" (replaces 1.1, 1.2, etc.)
     * Second-level numerical subclauses: "SUB_SUB_NUMBERED_ITEM: [content]" (replaces 1.1.1, 1.1.2, etc.)
     
     * alphabetical used in main level: "ALPHA_ITEM_MAIN: [content]" (replaces (a), (b), (c), etc.)
     * alphabetical used in first level subclauses: "SUB_ALPHA_ITEM: [content]" 
     * alphabetical used in second level subclauses: "SUB_SUB_ALPHA_ITEM: [content]"

     * roman numerals used in main level: "ROMAN_ITEM_MAIN: [content]" (replaces (i), (ii), (iii), etc.)
     * roman numerals used in first level subclauses: "SUB_ROMAN_ITEM: [content]"
     * roman numerals used in second level subclauses: "SUB_SUB_ROMAN_ITEM: [content]"
   
   - For bulleted lists (Either dependent on other lists or standalone):
     * Main bullet points: "BULLET_ITEM: [content]"
     * Sub-bullet points: "SUB_BULLET_ITEM: [content]"
     * Sub-sub bullet points: "SUB_SUB_BULLET_ITEM: [content]"
   
   - PRESERVE HIERARCHICAL RELATIONSHIPS:
     * Maintain the exact same hierarchy and structure as the original document
     * Ensure parent-child relationships between clauses are preserved
     * Keep cross-references intact but update to use the new prefix format
     * The PREFIX must exactly match one of the options above - no variations allowed
   
   - USE THESE FORMATTINGS ONLY WHILE WRITING CLAUSE LISTS. WRITE OTHER PARAGRAPHS AND SECTIONS IN REGULAR MARKDOWN FORMAT.

5. CLAUSE REQUIREMENTS:
   - Include ALL clauses from the original document in the template (if the clause is not relevant to the template, do not include it)
   - Maintain the exact structure and hierarchy of clauses using the PREFIX formats in point 4.
   - Keep the original clause organization
   - Preserve all legal definitions and references
   - Include all schedules, annexes, and appendices
   - Maintain cross-references between clauses
   - Keep all boilerplate language and standard provisions
   - Preserve the exact wording of legal terms and conditions
   - Include all jurisdictional and governing law clauses
   - Maintain all signature blocks and execution provisions
   - PRESERVE the existing clause hierarchy and numbering structure of the original document using the PREFIX formats in point 4.

Contract text:
{contract_text}"""
        
        try:
            self.logger.debug("Sending request to Gemini for initial template generation")
            template = self._call_gemini(prompt)
            
            self.logger.info("Successfully generated initial template")
            return template
            
        except Exception as e:
            self.logger.error(f"Failed to generate initial template: {str(e)}", exc_info=True)
            raise
    
    def update_template(self, current_template: str, new_contract: str) -> str:
        """Update template with new contract and enhanced placeholders"""
        self.logger.info("Updating template with new contract")
        
        if not self.api_config.is_configured():
            self.logger.error("API not configured")
            raise ValueError("API not configured")
        
        prompt = f"""You are an experienced solicitor in Hong Kong. You have decades of experience and are the senior partner of an international law firm. Here is a second legal document that you think is well drafted. You are trying to make a template using legal documents so you can reuse it in the future easily. You initially generated a template using a legal document you recently drafted.
        Below you are given both the current template and also a New contract document. Update your template based on this second document. 

REQUIREMENTS:
1. Merge the best practices from both documents
2. Add new clauses to the template using the second document where appropriate
3. Improve existing clauses
4. Ensure ALL placeholders use square brackets [Placeholder Name]
5. Identify and add any new placeholder fields from this contract
6. Maintain comprehensive placeholder coverage for all variable information

7. CLAUSE LIST FORMATTING REQUIREMENTS FOR AUTOMATED PROCESSING AND WORD COMPATIBILITY:
   - ALL numbered items MUST use the following prefix formats - DO NOT use manual numbering like "1." or "1.1" directly:
     * Main clauses: "NUMBERED_LIST_ITEM: [content]" (replaces 1., 2., 3., etc.)
     * First-level numerical subclauses: "SUB_NUMBERED_ITEM: [content]" (replaces 1.1, 1.2, etc.)
     * Second-level numerical subclauses: "SUB_SUB_NUMBERED_ITEM: [content]" (replaces 1.1.1, 1.1.2, etc.)
     
     * alphabetical used in main level: "ALPHA_ITEM_MAIN: [content]" (replaces (a), (b), (c), etc.)
     * alphabetical used in first level subclauses: "SUB_ALPHA_ITEM: [content]" 
     * alphabetical used in second level subclauses: "SUB_SUB_ALPHA_ITEM: [content]"

     * roman numerals used in main level: "ROMAN_ITEM_MAIN: [content]" (replaces (i), (ii), (iii), etc.)
     * roman numerals used in first level subclauses: "SUB_ROMAN_ITEM: [content]"
     * roman numerals used in second level subclauses: "SUB_SUB_ROMAN_ITEM: [content]"
   
   - For bulleted lists (Either dependent on other lists or standalone):
     * Main bullet points: "BULLET_ITEM: [content]"
     * Sub-bullet points: "SUB_BULLET_ITEM: [content]"
     * Sub-sub bullet points: "SUB_SUB_BULLET_ITEM: [content]"
   
   - PRESERVE HIERARCHICAL RELATIONSHIPS:
     * Maintain the exact same hierarchy and structure as the original document using the PREFIX formats in point 7.
     * Ensure parent-child relationships between clauses are preserved
     * Keep cross-references intact but update to use the new prefix format
     * The PREFIX must exactly match one of the options above - no variations allowed
   
   - USE THESE FORMATTINGS ONLY WHILE WRITING CLAUSE LISTS. WRITE OTHER PARAGRAPHS AND SECTIONS IN REGULAR MARKDOWN FORMAT.

   
8. CLAUSE STRUCTURE AND HIERARCHY REQUIREMENTS:
   - Maintain a logical and consistent clause hierarchy throughout the template
   - Use proper hierarchical numbering for clauses and subclauses following the PREFIX FORMAT in point 7:
     * Main clauses: 1., 2., 3. (use PREFIX "NUMBERED_LIST_ITEM:")
     * First level subclauses: 1.1, 1.2, 1.3 (use PREFIX "SUB_NUMBERED_ITEM:")
     * Second level subclauses: 1.1.1, 1.1.2, 1.1.3 (use PREFIX "SUB_SUB_NUMBERED_ITEM:") and so on as mentioned in point 7
   - Group related provisions into subclauses under a common parent clause
   - Ensure logical flow between parent clauses and their subclauses
   - When merging documents with different hierarchical structures, standardize the approach
   - Create subclauses when a clause has multiple related concepts that deserve separate treatment
   - Ensure cross-references in the document correctly reflect the hierarchical structure
   - Maintain consistent indentation for each level of the hierarchy
   - If either document contains well-structured subclauses, preserve and enhance this structure
   - IMPORTANT: The structure relationships must be preserved, but ALL numbering must use the PREFIX formats in point 7

9. CLAUSE REQUIREMENTS:
   - Include ALL clauses from the new contract (if the clause is not relevant to the template, do not include it)
   - Maintain the exact structure and hierarchy of clauses
   - Keep the original clause organization
   - Preserve all legal definitions and references
   - Include all schedules, annexes, and appendices
   - Maintain cross-references between clauses
   - Keep all boilerplate language and standard provisions
   - Preserve the exact wording of legal terms and conditions
   - Include all jurisdictional and governing law clauses
   - Maintain all signature blocks and execution provisions

Current template:
{current_template}

New contract:
{new_contract}"""
        
        try:
            self.logger.debug("Sending request to Gemini for template update")
            updated_template = self._call_gemini(prompt)
            
            self.logger.info("Successfully updated template")
            return updated_template
            
        except Exception as e:
            self.logger.error(f"Failed to update template: {str(e)}", exc_info=True)
            raise
    
    def add_drafting_notes(self, template_text: str, all_contracts: List[Dict]) -> str:
        """Add comprehensive drafting notes with alternatives from uploaded contracts"""
        self.logger.info("Adding enhanced drafting notes to template")
        
        if not self.api_config.is_configured():
            self.logger.error("API not configured")
            raise ValueError("API not configured")
        
        # Prepare context from all uploaded contracts for alternative clauses
        contracts_context = "\n\n".join([
            f"CONTRACT: {contract['metadata'].get('contract_name', contract.get('filename', 'Unknown'))}\n"
            f"PARTIES: {', '.join(contract['metadata'].get('parties', []))}\n"
            f"TEXT EXCERPT: {contract['extracted_text'][:1000]}..."
            for contract in all_contracts
        ])
        
        prompt = f"""Using your legal documents, You have created a template for your junior lawyers. Now add comprehensive drafting notes to the template so that your junior lawyers can know when and how to use the template effectively. 
        Also, keep in mind this is going to be the final version of the template that is going to be used in publication, so in your response only output the template text with the enhancement of drafting notes and alternatives. 
        Also, ensure the template text in your response is complete as a legal document in accordance with the uploaded contract document or documents, just with added drafting notes and alternative clauses.

REQUIREMENTS FOR DRAFTING NOTES:
1. For each clause, add a drafting note immediately after the clause number/heading that explains:
   - Where this clause should be placed in the template (e.g., "This clause should follow Section 1.1 and precede Section 1.2")
   - The exact meaning and purpose of the clause
   - When to use this clause
   - Legal implications
   - Example: 
     "1. Definitions 
     [DRAFTING NOTE: This clause should be placed at the beginning of the agreement, after the recitals. It defines key terms used throughout the document. Essential for all contracts involving [specific scenario]. Ensures clarity in interpretation and prevents disputes over terminology.]"
     [Full clause text here...] 

2. Format notes as: [DRAFTING NOTE: your detailed note here]

REQUIREMENTS FOR ALTERNATIVE CLAUSES:
1. For each clause that has alternatives, add them immediately after the clause with proper numbering and drafting notes explaining the usage of the alternative:
   Example:
   "16. Governing Law and Jurisdiction
   [DRAFTING NOTE: ...]
   [FULL CLAUSE TEXT HERE]
   
   [ALTERNATIVE CLAUSE: Alternative Clause: Arbitration Clause - Use when: Confidentiality and potentially faster resolution are preferred over public court proceedings.]
   [DRAFTING NOTE: Arbitration is a form of alternative dispute resolution (ADR). It can be faster, more flexible, and is confidential. However, it can also be expensive, and appeal rights are very limited. This is a significant strategic choice. For B2C contracts, forcing arbitration can sometimes be viewed unfavourably by consumers or regulators. This clause should specify the arbitration institution (e.g., HKIAC), the location, language, and number of arbitrators.]"

2. For each alternative, explain:
   - When it should be used instead
   - Legal implications of each variation
   - Industry-specific considerations
   - Risk assessment
   - Compliance implications
   - Best practices for selection

3. Format alternatives as: [ALTERNATIVE CLAUSE: alternative formulation - Use when: specific scenario]

4. CLAUSE LIST FORMATTING REQUIREMENTS FOR AUTOMATED PROCESSING AND WORD COMPATIBILITY:
   - ALL numbered items MUST use the following prefix formats - DO NOT use manual numbering like "1." or "1.1" directly:
     * Main clauses: "NUMBERED_LIST_ITEM: [content]" (replaces 1., 2., 3., etc.)
     * First-level numerical subclauses: "SUB_NUMBERED_ITEM: [content]" (replaces 1.1, 1.2, etc.)
     * Second-level numerical subclauses: "SUB_SUB_NUMBERED_ITEM: [content]" (replaces 1.1.1, 1.1.2, etc.)
     
     * alphabetical used in main level: "ALPHA_ITEM_MAIN: [content]" (replaces (a), (b), (c), etc.)
     * alphabetical used in first level subclauses: "SUB_ALPHA_ITEM: [content]" 
     * alphabetical used in second level subclauses: "SUB_SUB_ALPHA_ITEM: [content]"

     * roman numerals used in main level: "ROMAN_ITEM_MAIN: [content]" (replaces (i), (ii), (iii), etc.)
     * roman numerals used in first level subclauses: "SUB_ROMAN_ITEM: [content]"
     * roman numerals used in second level subclauses: "SUB_SUB_ROMAN_ITEM: [content]"
   
   - For bulleted lists (Either dependent on other lists or standalone):
     * Main bullet points: "BULLET_ITEM: [content]"
     * Sub-bullet points: "SUB_BULLET_ITEM: [content]"
     * Sub-sub bullet points: "SUB_SUB_BULLET_ITEM: [content]"
   
   - PRESERVE HIERARCHICAL RELATIONSHIPS:
     * Maintain the exact same hierarchy and structure as the original document
     * Ensure parent-child relationships between clauses are preserved
     * Keep cross-references intact but update to use the new prefix format
     * The PREFIX must exactly match one of the options above - no variations allowed
   
   - USE THESE FORMATTINGS ONLY WHILE WRITING CLAUSE LISTS. WRITE OTHER PARAGRAPHS AND SECTIONS IN REGULAR MARKDOWN FORMAT.

5. CLAUSE RELEVANCE ASSESSMENT:
   - For each clause, provide clear guidance on when it should be included or excluded
   - Consider industry-specific requirements
   - Account for different contract types and purposes
   - Consider jurisdictional variations
   - Include risk-based assessment
   - Provide clear decision criteria

UPLOADED CONTRACTS FOR REFERENCE:
{contracts_context}

Template to enhance:
{template_text}"""
        
        try:
            self.logger.debug("Sending request to Gemini for enhanced drafting notes")
            template_with_notes = self._call_gemini(prompt)
            
            self.logger.info("Successfully added enhanced drafting notes")
            return template_with_notes
            
        except Exception as e:
            self.logger.error(f"Failed to add drafting notes: {str(e)}", exc_info=True)
            raise
    
    def extract_metadata(self, contract_text: str) -> Dict[str, Any]:
        """Extract metadata from contract text including contract name, parties, and date"""
        self.logger.info("Extracting contract metadata")
        
        if not self.api_config.is_configured():
            self.logger.error("API not configured")
            raise ValueError("API not configured")
        
        prompt = f"""Suppose, You are an experienced solicitor in Hong Kong. You have decades of experience and are the senior partner of an international law firm.
Extract the key metadata from this legal document. Please identify:
1. The contract name/title
2. The parties involved (as a list)
3. The contract date
4. The governing law
5. The contract type (e.g., employment, lease, service agreement, etc.)

REQUIREMENTS:
- Return the data in a structured JSON format
- For parties, list all entities involved in the contract
- For dates, use YYYY-MM-DD format when possible
- If information is not found, use null

Here's the document:

{contract_text[:8000]}  
"""
        
        try:
            response = self._call_gemini(
                prompt=prompt,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            metadata = json.loads(response)
            
            # Ensure expected fields exist
            default_metadata = {
                "contract_name": "Unknown Contract",
                "parties": [],
                "date": None,
                "governing_law": None,
                "contract_type": "Unknown"
            }
            
            # Merge with defaults
            for key, default_value in default_metadata.items():
                if key not in metadata or metadata[key] is None:
                    metadata[key] = default_value
            
            return metadata
            
        except json.JSONDecodeError:
            self.logger.error("Failed to parse metadata JSON response")
            return {
                "contract_name": "Unknown Contract",
                "parties": [],
                "date": None,
                "governing_law": None,
                "contract_type": "Unknown"
            }
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {str(e)}", exc_info=True)
            return {
                "contract_name": "Unknown Contract",
                "parties": [],
                "date": None,
                "governing_law": None,
                "contract_type": "Unknown"
            }
        

    def extract_used_clauses(self, template_content: str) -> List[Dict[str, Any]]:
        """
        Extract clauses that are actually used in the generated template content.
        Returns clauses in the same format used in the clause_library.
        """
        self.logger.info("🔍 Starting extraction of clauses from generated template")
        
        if not self.api_config.is_configured():
            self.logger.error("❌ API not configured for clause extraction")
            raise ValueError("API not configured")
        
        if not template_content or len(template_content.strip()) == 0:
            self.logger.warning("❌ Empty template content provided")
            return []
        
        # Truncate very long content to prevent API limits
        max_content_length = 100000  # 100k characters
        if len(template_content) > max_content_length:
            self.logger.warning(f"⚠️  Template content too long ({len(template_content)} chars), truncating to {max_content_length}")
            template_content = template_content[:max_content_length] + "\n\n[Content truncated due to length]"

        prompt = f"""You are an experienced legal analyst with expertise in contract clause identification and classification.
        
TASK:
Extract all legal clauses present in the provided template content and categorize them in the same format used for the clause library.

For each clause, identify:
1. Clause type (e.g., definitions, payment, termination, liability, intellectual_property, confidentiality, dispute_resolution, governing_law, force_majeure, etc.)
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

IMPORTANT INSTRUCTIONS:
- Identify ALL distinct clauses in the template
- Maintain the EXACT text of each clause (including placeholders in square brackets)
- Each clause should be categorized by type
- Ensure every important legal provision is captured
- Do not omit any clauses, even if they seem standard
- Ensure the output matches the required schema format exactly and is valid JSON
- Include comprehensive relevance assessment for each clause
- Ensure that the order of clauses matches their appearance in the template and that the hierarchy is preserved.

IMPORTANT: Inside any field that contains text from the contract (like "clause_text"), you MUST escape any double quotes (") with a backslash (\\"). For example, if the text is 'The term "Agreement"...', it must be represented in the JSON as '"clause_text": "The term \\"Agreement\\"..."'.


Return the result as a valid JSON object with a "clauses" array containing objects with the following structure:
{{
  "clauses": [
    {{
      "clause_type": "payment",
      "clause_text": "complete exact clause text with placeholders preserved",
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

Template content:
{template_content}"""
        
        try:
            self.logger.debug(f"📤 Sending request to extract clauses from template (content length: {len(template_content)})")
            response = self.api_config.generate_with_schema(
                prompt=prompt,
                schema_class=ClauseExtraction,
                temperature=0.1
            )
            
            if not response or len(response.strip()) == 0:
                self.logger.warning("❌ Empty response from API when extracting clauses")
                return []
            
            self.logger.debug(f"📥 Received response from API (length: {len(response)})")
            
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
                    repaired_json = self._repair_json_response(response)
                    
                    if not repaired_json:
                        self.logger.error("❌ Failed to repair JSON response")
                        return []
                    
                    extracted_data = json.loads(repaired_json)
                    clauses = extracted_data.get("clauses", [])
                    
                    if not isinstance(clauses, list):
                        self.logger.error("❌ Repaired JSON still does not contain a valid clauses list")
                        return []
                
                # Add timestamp and source info to each clause
                for clause in clauses:
                    clause["extracted_at"] = datetime.utcnow().isoformat()
                    clause["source_contract"] = "Generated Template" 
                    clause["contract_filename"] = "Generated Template"
                    clause["contract_parties"] = []
                
                self.logger.info(f"✅ Successfully extracted {len(clauses)} clauses from the template")
                
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
            self.logger.error(f"❌ Failed to extract clauses from template: {str(e)}", exc_info=True)
            return []
    
    def _repair_json_response(self, invalid_json: str) -> str:
        """Use a second LLM call to repair invalid JSON responses"""
        # First, try to identify the specific error
        error_message = "Unknown JSON error"
        try:
            json.loads(invalid_json)
        except json.JSONDecodeError as e:
            error_message = str(e)
        
        repair_prompt = f"""You are a JSON repair expert. The following text is a JSON response that has validation errors or is malformed.
Your task is to fix the JSON and return a properly formatted valid JSON object.

6. Incorrect nesting of objects or arrays

The JSON should have the following structure:
{{
  "clauses": [
    {{
      "clause_type": "string",
      "clause_text": "string",
      "position_context": "string",
      "clause_purpose": "string",
      "relevance_assessment": {{
        "when_to_include": ["string"],
        "when_to_exclude": ["string"],
        "industry_considerations": ["string"],
        "risk_implications": ["string"],
        "compliance_requirements": ["string"],
        "best_practices": ["string"]
      }}
    }}
  ]
}}

IMPORTANT: Inside any field that contains text from the contract (like "clause_text"), you MUST escape any double quotes (") with a backslash (\\"). For example, if the text is 'The term "Agreement"...', it must be represented in the JSON as '"clause_text": "The term \\"Agreement\\"..."'.

Do not add any explanations or comments, just return the fixed JSON.

Invalid JSON:
{invalid_json}"""

        try:
            self.logger.debug(f"Attempting to repair invalid JSON with second LLM call. Error: {error_message}")
            repaired_response = self.api_config.generate_text(
                prompt=repair_prompt,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Validate that the repaired JSON is valid
            try:
                json.loads(repaired_response)
                self.logger.info("Successfully repaired JSON response")
                return repaired_response
            except json.JSONDecodeError as e:
                self.logger.error(f"Repair attempt failed to produce valid JSON: {e}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Failed to repair JSON: {str(e)}", exc_info=True)
            return ""

    # async def generate_template(self, template_data: Dict[str, Any]) -> str:
    #     """Generate template from processed files data"""
    #     try:
    #         self.logger.info(f"Generating template: {template_data.get('name', 'Unnamed')}")
            
    #         files = template_data.get('files', [])
    #         priority_file_id = template_data.get('priority_file_id')
    #         folder_id = template_data.get('folder_id')
    #         template_name = template_data.get('name', f"Generated Template - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
    #         if not files:
    #             raise ValueError("No files provided for template generation")
            
    #         # Get markdown content for each file
    #         from core.database import get_database_service
    #         db_service = get_database_service()
            
    #         processed_files = []
    #         for file_data in files:
    #             file_id = file_data.get('id')
    #             if not file_id:
    #                 continue
                    
    #             # Get markdown content
    #             markdown_response = db_service.client.table("markdown_content").select("*").eq("file_id", file_id).execute()
    #             if markdown_response.data:
    #                 markdown_content = markdown_response.data[0]['content']
    #                 processed_files.append({
    #                     'file_id': file_id,
    #                     'filename': file_data.get('original_filename', 'Unknown'),
    #                     'extracted_text': markdown_content
    #                 })
            
    #         if not processed_files:
    #             raise ValueError("No processed files with content available")
            
    #         # Reorder files to put priority file first
    #         if priority_file_id:
    #             priority_file = next((f for f in processed_files if f['file_id'] == priority_file_id), None)
    #             if priority_file:
    #                 processed_files.remove(priority_file)
    #                 processed_files.insert(0, priority_file)
            
    #         # Generate template content
    #         if len(processed_files) == 1:
    #             template_content = self.generate_initial_template(processed_files[0]['extracted_text'])
    #         else:
    #             # Start with first file
    #             template_content = self.generate_initial_template(processed_files[0]['extracted_text'])
                
    #             # Update with additional files
    #             for contract in processed_files[1:]:
    #                 template_content = self.update_template(template_content, contract['extracted_text'])
            
    #         # Add drafting notes
    #         final_template = self.add_drafting_notes(template_content, processed_files)
            
    #         # Save template to database
    #         template_data_db = {
    #             "folder_id": folder_id,
    #             "name": template_name,
    #             "content": final_template,
    #             "template_type": "general",
    #             "file_extension": ".docx",
    #             "formatting_data": {
    #                 "source_files": [f['filename'] for f in processed_files],
    #                 "generation_date": datetime.utcnow().isoformat(),
    #                 "ai_generated": True,
    #                 "priority_file": priority_file_id
    #             },
    #             "word_compatible": True,
    #             "is_active": True
    #         }
            
    #         template_result = db_service.client.table("templates").insert(template_data_db).execute()
            
    #         if template_result.data:
    #             template_id = template_result.data[0]["id"]
    #             self.logger.info(f"Successfully generated template {template_id}")
    #             return template_id
    #         else:
    #             raise ValueError("Failed to save template to database")
                
    #     except Exception as e:
    #         self.logger.error(f"Template generation failed: {str(e)}", exc_info=True)
    #         raise