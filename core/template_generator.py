import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from core.api_config import APIConfiguration
from pydantic import BaseModel, Field
from enum import Enum  # This is missing

logger.add("logs/template_generator.log", rotation="100 MB", level="DEBUG", backtrace=True, diagnose=True)



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




class AlternativeClause(BaseModel):
    alternative_text: str
    drafting_note: str
    use_when: str

class SubClause(BaseModel):
    subclause_number: Union[str, float]  # Allow both numeric and text identifiers
    subclause_text: str
    drafting_note: str
    alternatives: Optional[List[AlternativeClause]] = Field(default_factory=list)

class MainClause(BaseModel):
    type: ClauseType
    clause_text: str
    drafting_note: str
    subclauses: Optional[List[SubClause]] = Field(default_factory=list)  # Changed from Dict to List
    alternatives: Optional[List[AlternativeClause]] = Field(default_factory=list)

class ContractTemplate(BaseModel):
    general_text: str
    clauses: List[MainClause]  # Changed from Dict to List to avoid $ref issues with Gemini API

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
        
        prompt = f"""You are an experienced solicitor in Hong Kong with decades of experience and are the senior partner of an international law firm. You are creating a comprehensive legal document template for future reuse.

**TASK:** Update your existing template by incorporating a new contract document, merging the best practices from both documents.

**INPUTS:**
- Current template: {current_template}
- New contract: {new_contract}

**FORMATTING REQUIREMENTS:**

**1. PLACEHOLDER SYSTEM**
- ALL variable information MUST use square brackets: [Placeholder Name]
- Examples: [Company Name], [Contract Date], [Governing Law], [Principal Amount]
- Ensure comprehensive placeholder coverage for all variable information

**2. CLAUSE NUMBERING SYSTEM (MANDATORY - For Word/Automated Processing Compatibility)**

**Exact Prefix Formats Required:**
- Main clauses: `1.`, `2.`, `3.`, `4.`
- First level subclauses: `1.1`, `1.2`, `1.3`, `2.1`, `2.2`
- Second level subclauses: `1.1.1`, `1.1.2`, `1.1.3`, `2.1.1`, `2.1.2`
- Continue this pattern for deeper levels: `1.1.1.1`, `1.1.1.2`, etc.

**Apply Clause Numbering ONLY to:**
- Legal clauses and provisions
- Terms and conditions
- Numbered legal requirements

**Use Regular Markdown for:**
- Document headers and titles
- Introductory paragraphs
- Signature blocks
- Recitals and whereas clauses

**3. HIERARCHICAL STRUCTURE REQUIREMENTS**
- Preserve parent-child relationships between clauses
- Maintain logical flow from main clauses to subclauses
- Group related provisions under common parent clauses
- Update cross-references to match new numbering format
- Use consistent indentation for hierarchy levels
- When merging different structures, standardize using the above format

**4. CONTENT INTEGRATION REQUIREMENTS**
- Merge best practices from both documents
- Add new relevant clauses from the second document
- Improve existing clauses where appropriate
- Include ALL relevant clauses (exclude only those not applicable to template)
- Preserve exact legal definitions, references, and boilerplate language
- Maintain all schedules, annexes, and appendices
- Keep jurisdictional and governing law clauses
- Preserve signature blocks and execution provisions
- Maintain all cross-references between clauses

**5. QUALITY STANDARDS**
- Ensure legal precision and accuracy
- Maintain professional drafting standards
- Preserve Hong Kong legal conventions
- Keep international law firm quality standards

**OUTPUT:** A comprehensive, well-structured legal document template ready for future use with proper placeholder coverage and standardized formatting."""
        
        try:
            self.logger.debug("Sending request to Gemini for template update")
            updated_template = self._call_gemini(prompt)
            
            self.logger.info("Successfully updated template")
            return updated_template
            
        except Exception as e:
            self.logger.error(f"Failed to update template: {str(e)}", exc_info=True)
            raise
    

    def add_drafting_notes(self, template_text: str) -> str:
        """Add comprehensive drafting notes with alternatives from uploaded contracts"""
        self.logger.info("Adding enhanced drafting notes to template")
        
        if not self.api_config.is_configured():
            self.logger.error("API not configured")
            raise ValueError("API not configured")
        
        # # Prepare context from all uploaded contracts for alternative clauses
        # contracts_context = "\n\n".join([
        #     f"CONTRACT: {contract['metadata'].get('contract_name', contract.get('filename', 'Unknown'))}\n"
        #     f"PARTIES: {', '.join(contract['metadata'].get('parties', []))}\n"
        #     f"TEXT EXCERPT: {contract['extracted_text'][:1000]}..."
        #     for contract in all_contracts
        # ])
        
        prompt = f"""Using your legal documents, you have created a template for your junior lawyers. Now add comprehensive drafting notes to the template so that your junior lawyers can know when and how to use the template effectively. You shouldn't change anything in the clauses, I want to have all of them in the following format.

You must return the response ONLY in JSON format following this exact structure:

    {{
        "general_sections": {{
            "preamble": "Any introductory text that appears before the numbered clauses",
            "recitals": "WHEREAS clauses and background information",
            "definitions": "Key definitions section if not included in numbered clauses",
            "signature_block": "Template for signature and execution section"
        }},
        "clauses": [
            {{
                "clause_number": "1.",
                "clause_type": "Payment Terms Clause",
                "clause_title": "[Title of the Clause]",
                "clause_text": "Full text of the main clause with all [Placeholders] intact...",
                "drafting_note": "Detailed explanation of when and how to use this clause, its purpose, legal implications, and placement guidance...",
                "subclauses": [
                    {{
                        "subclause_number": "1.1",
                        "subclause_text": "Full text of subclause 1.1 with [Placeholders]...",
                        "drafting_note": "Explanation of this subclause's specific function, legal implications, and practical advice for customizing...",
                        "sub_subclauses": [
                            {{
                                "sub_subclause_number": "1.1.1",
                                "sub_subclause_text": "Full text of sub-subclause 1.1.1 with [Placeholders]...",
                                "drafting_note": "Explanation of this sub-subclause's specific function, legal implications, and practical advice...",
                                "sub_sub_subclauses": [
                                    {{
                                        "sub_sub_subclause_number": "1.1.1.1",
                                        "sub_sub_subclause_text": "Full text of sub-sub-subclause 1.1.1.1 with [Placeholders]...",
                                        "drafting_note": "Explanation of this provision, legal implications, and implementation advice...",
                                        "alternatives": [
                                            {{
                                                "alternative_text": "Alternative wording...",
                                                "drafting_note": "Explanation of this alternative including legal implications and risk assessment...",
                                                "use_when": "When this alternative should be used"
                                            }}
                                        ]
                                    }}
                                ],
                                "alternatives": [
                                    {{
                                        "alternative_text": "Alternative wording for this sub-subclause...",
                                        "drafting_note": "Explanation of this alternative including legal implications and risk assessment...",
                                        "use_when": "When this alternative should be used"
                                    }}
                                ]
                            }}
                        ],
                        "alternatives": [
                            {{
                                "alternative_text": "Alternative wording for this subclause...",
                                "drafting_note": "Explanation of this alternative including legal implications, industry considerations, and risk assessment...",
                                "use_when": "Specific scenario when this alternative should be used"
                            }}
                        ]
                    }}
                ],
                "alternatives": [
                    {{
                        "alternative_text": "Alternative wording for the entire main clause...",
                        "drafting_note": "Detailed explanation of this alternative including legal implications, industry considerations, and risk assessment...",
                        "use_when": "Specific scenario when this alternative should be used instead of the main clause..."
                    }}
                ]
            }},
            {{
                "clause_number": "2.",
                "clause_type": "Scope of Work (Statement of Work) Clause",
                "clause_title": "[Title of Clause 2]",
                "clause_text": "Full text of clause 2 with [Placeholders]...",
                "drafting_note": "Detailed explanation of purpose, positioning advice, usage scenarios, legal implications, implementation guidelines, and risk considerations..."
            }}
        ]
    }}

**REQUIREMENTS FOR DRAFTING NOTES:**

1. **For each clause, the drafting_note must explain:**
   - The exact legal meaning and purpose of the clause
   - Where this clause should be placed in the document
   - When to use this clause vs alternatives
   - Legal implications and consequences
   - Best practices for implementation
   - Risk assessment and mitigation strategies

2. **For alternative clauses, explain:**
   - When it should be used instead of the main clause
   - Legal implications of each variation
   - Industry-specific considerations
   - Risk assessment compared to main clause
   - Compliance implications

3. **CLAUSE TYPES must be selected from this list:**
   - **Fundamental Clauses:** Payment Terms Clause, Scope of Work (Statement of Work) Clause, Term & Termination Clause, Price Adjustment / Escalation Clause
   - **Protective Clauses:** Indemnity Clause, Limitation of Liability Clause, Exemption / Exclusion Clause, Liquidated Damages Clause, Exculpatory Clause, Gross-Up Clause, Retention of Title (Romalpa) Clause
   - **Dispute Resolution Clauses:** Arbitration Clause, Dispute Resolution / Escalation Clause, Choice of Law Clause, Confession of Judgment Clause
   - **Confidentiality & IP Clauses:** Confidentiality / Non-Disclosure Clause, Intellectual Property Clause, Non-Compete Clause, Non-Solicitation Clause
   - **Operational (Boilerplate) Clauses:** Assignment Clause, Change Control / Changes Clause, Amendment Clause, Notice Clause, Severability Clause, Survival Clause, Entire Agreement Clause, Waiver Clause, Interpretation Clause, Electronic Signatures Clause

4. **Ensure the template includes:**
   - All necessary clauses from the uploaded contracts
   - Complete legal document structure
   - Logical clause ordering
   - Comprehensive coverage of all legal aspects
   - All placeholders properly identified

5. **Placeholder Management:**
   - Maintain ALL existing [Placeholder] formats
   - Document all required placeholders with descriptions
   - Specify data types and requirements
   - Provide completion guidance

**Template to enhance (follow exactly, do not change clauses):**
{template_text}

**IMPORTANT:** Return ONLY the JSON response. Do not include any other text or explanation outside the JSON structure"""
        
        try:
            self.logger.debug("Sending request to Gemini for enhanced drafting notes (schema mode)")
            try:
                template_with_notes = self.api_config.generate_text(
                    prompt=prompt,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
            except Exception as schema_err:
                # Log and fallback to simpler JSON generation without schema
                self.logger.error(f"Schema-based generation failed: {schema_err}. Falling back to simple JSON generation")
                template_with_notes = self.api_config.generate_text(
                    prompt=prompt,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

            # Validate that the response is valid JSON
            try:
                json.loads(template_with_notes)
                self.logger.info("Successfully added enhanced drafting notes")
                return template_with_notes
            except json.JSONDecodeError as json_error:
                self.logger.warning(f"Initial response was invalid JSON: {str(json_error)}, attempting repair")
                repaired_response = self.repair_json_response_template(template_with_notes)
                if repaired_response:
                    return repaired_response
                else:
                    self.logger.error("Failed to repair JSON response, returning original template")
                    return template_text
            
        except Exception as e:
            self.logger.error("Failed to add drafting notes: {}", str(e), exc_info=True)
            raise

    def repair_json_response_template(self, invalid_json: str) -> str:
        """Use a second LLM call to repair invalid JSON responses"""
        # First, try to identify the specific error
        error_message = "Unknown JSON error"
        try:
            json.loads(invalid_json)
        except json.JSONDecodeError as e:
            error_message = str(e)
            
        repair_prompt = f"""You are a JSON repair expert. The following text is a JSON response that has validation errors or is malformed.
    Your task is to fix the JSON and return a properly formatted valid JSON object.

    Common JSON errors to fix:
    1. Missing quotes around keys or values
    2. Trailing commas
    3. Unescaped quotes within strings
    4. Missing commas between objects/arrays
    5. Mismatched brackets or braces
    6. Incorrect nesting of objects or arrays

    The JSON should have the following structure:
    {{
        "general_sections": {{
            "preamble": "Any introductory text that appears before the numbered clauses",
            "recitals": "WHEREAS clauses and background information",
            "definitions": "Key definitions section if not included in numbered clauses",
            "signature_block": "Template for signature and execution section"
        }},
        "clauses": [
            {{
                "clause_number": "1.",
                "clause_type": "Payment Terms Clause",
                "clause_title": "[Title of the Clause]",
                "clause_text": "Full text of the main clause with all [Placeholders] intact...",
                "drafting_note": "Detailed explanation of when and how to use this clause, its purpose, legal implications, and placement guidance of every clauses in this...",
                "subclauses": [
                    {{
                        "subclause_number": "1.1",
                        "subclause_text": "Full text of subclause 1.1 with [Placeholders]...",
                        "drafting_note": "Explanation of this subclause's specific function, legal implications, and practical advice for customizing...",
                        "sub_subclauses": [
                            {{
                                "sub_subclause_number": "1.1.1",
                                "sub_subclause_text": "Full text of sub-subclause 1.1.1 with [Placeholders]...",
                                "drafting_note": "Explanation of this sub-subclause's specific function, legal implications, and practical advice...",
                                "sub_sub_subclauses": [
                                    {{
                                        "sub_sub_subclause_number": "1.1.1.1",
                                        "sub_sub_subclause_text": "Full text of sub-sub-subclause 1.1.1.1 with [Placeholders]...",
                                        "drafting_note": "Explanation of this provision, legal implications, and implementation advice...",
                                        "alternatives": [
                                            {{
                                                "alternative_text": "Alternative wording...",
                                                "drafting_note": "Explanation of this alternative including legal implications and risk assessment...",
                                                "use_when": "When this alternative should be used"
                                            }}
                                        ]
                                    }}
                                ],
                                "alternatives": [
                                    {{
                                        "alternative_text": "Alternative wording for this sub-subclause...",
                                        "drafting_note": "Explanation of this alternative including legal implications and risk assessment...",
                                        "use_when": "When this alternative should be used"
                                    }}
                                ]
                            }}
                        ],
                        "alternatives": [
                            {{
                                "alternative_text": "Alternative wording for this subclause...",
                                "drafting_note": "Explanation of this alternative including legal implications, industry considerations, and risk assessment...",
                                "use_when": "Specific scenario when this alternative should be used"
                            }}
                        ]
                    }}
                ],
                "alternatives": [
                    {{
                        "alternative_text": "Alternative wording for the entire main clause...",
                        "drafting_note": "Detailed explanation of this alternative including legal implications, industry considerations, and risk assessment...",
                        "use_when": "Specific scenario when this alternative should be used instead of the main clause..."
                    }}
                ]
            }},
            {{
                "clause_number": "2.",
                "clause_type": "Scope of Work (Statement of Work) Clause",
                "clause_title": "[Title of Clause 2]",
                "clause_text": "Full text of clause 2 with [Placeholders]...",
                "drafting_note": "Detailed explanation of purpose, positioning advice, usage scenarios, legal implications, implementation guidelines, and risk considerations..."
            }}
        ]
    }}

    PREDEFINED CLAUSE TYPES (clause_type field MUST be exactly one of these values):
    - Fundamental Clauses: "Payment Terms Clause", "Scope of Work (Statement of Work) Clause", "Term & Termination Clause", "Price Adjustment / Escalation Clause"
    - Protective Clauses: "Indemnity Clause", "Limitation of Liability Clause", "Exemption / Exclusion Clause", "Liquidated Damages Clause", "Exculpatory Clause", "Gross-Up Clause", "Retention of Title (Romalpa) Clause"
    - Dispute Resolution Clauses: "Arbitration Clause", "Dispute Resolution / Escalation Clause", "Choice of Law Clause", "Confession of Judgment Clause"
    - Confidentiality & IP Clauses: "Confidentiality / Non-Disclosure Clause", "Intellectual Property Clause", "Non-Compete Clause", "Non-Solicitation Clause"
    - Operational (Boilerplate) Clauses: "Assignment Clause", "Change Control / Changes Clause", "Amendment Clause", "Notice Clause", "Severability Clause", "Survival Clause", "Entire Agreement Clause", "Waiver Clause", "Interpretation Clause", "Electronic Signatures Clause"

    IMPORTANT RULES:
    1. Inside any field that contains text from the contract (like "clause_text", "drafting_note"), you MUST escape any double quotes (") with a backslash (\\"). For example, if the text is 'The term "Agreement"...', it must be represented in the JSON as '"clause_text": "The term \\"Agreement\\"..."'.
    2. The "clause_type" field must match exactly one of the predefined clause types listed above
    3. "subclauses", "sub_subclauses", "sub_sub_subclauses", and "alternatives" are optional fields
    4. Ensure proper nesting and comma placement
    5. Remove any trailing commas
    6. Maintain the hierarchical numbering system (1., 1.1, 1.1.1, 1.1.1.1)
    7. All [Placeholders] should remain in square brackets format
    8. Do not add any explanations or comments, just return the fixed JSON

    JSON Error Found: {error_message}

    Invalid JSON:
    {invalid_json}"""

        try:
            self.logger.debug("Sending JSON repair request to Gemini")
            repaired_json = self.api_config.generate_text(
                prompt=repair_prompt,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Validate the repaired JSON
            json.loads(repaired_json)
            self.logger.info("Successfully repaired JSON")
            return repaired_json
            
        except Exception as e:
            self.logger.error(f"Failed to repair JSON: {str(e)}", exc_info=True)
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

        prompt = f"""You are an experienced legal analyst with expertise in contract clause identification and classification.
        
TASK:
    Extract all legal clauses present in the provided template content and categorize them using the predefined clause types.

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
- Identify ALL distinct clauses in the template
- Maintain the EXACT text of each clause (including placeholders in square brackets)
    - Each clause MUST be categorized using one of the predefined clause types listed above
- Ensure every important legal provision is captured
- Do not omit any clauses, even if they seem standard
- Ensure the output matches the required schema format exactly and is valid JSON
- Include comprehensive relevance assessment for each clause
    - Ensure that the order of clauses matches their appearance in the template and that the hierarchy is preserved

IMPORTANT: Inside any field that contains text from the contract (like "clause_text"), you MUST escape any double quotes (") with a backslash (\\"). For example, if the text is 'The term "Agreement"...', it must be represented in the JSON as '"clause_text": "The term \\"Agreement\\"..."'.

Return the result as a valid JSON object with a "clauses" array containing objects with the following structure:
{{
  "clauses": [
    {{
        "clause_type": "Payment Terms Clause",
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
                    repaired_json = self._repair_json_response_clause_extraction(response)
                    
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
            self.logger.error(f"Failed to extract clauses: {str(e)}", exc_info=True)
            return []
    
    def _repair_json_response_clause_extraction(self, invalid_json: str) -> str:
        """Use a second LLM call to repair invalid JSON responses"""
        # First, try to identify the specific error
        error_message = "Unknown JSON error"
        try:
            json.loads(invalid_json)
        except json.JSONDecodeError as e:
            error_message = str(e)
        
        repair_prompt = f"""You are a JSON repair expert. The following text is a JSON response that has validation errors or is malformed.
Your task is to fix the JSON and return a properly formatted valid JSON object.

    Common JSON errors to fix:
    1. Missing quotes around keys or values
    2. Trailing commas
    3. Unescaped quotes within strings
    4. Missing commas between objects/arrays
    5. Mismatched brackets or braces
6. Incorrect nesting of objects or arrays

The JSON should have the following structure:
{{
        "general_text": "Any general introductory text or preamble that appears before the numbered clauses",
  "clauses": [
    {{
                "type": "Payment Terms Clause",
                "clause_text": "Full text of the main clause...",
                "drafting_note": "Detailed explanation of when and how to use this clause...",
                "subclauses": [
                    {{
                        "subclause_number": "1.1",
                        "subclause_text": "Full text of subclause 1.1...",
                        "drafting_note": "Explanation of this subclause...",
                        "alternatives": [
                            {{
                                "alternative_text": "Alternative wording for this subclause...",
                                "drafting_note": "Explanation of this alternative...",
                                "use_when": "Specific scenario when this alternative should be used..."
                            }}
                        ]
                    }},
                    {{
                        "subclause_number": "1.2",
                        "subclause_text": "Full text of subclause 1.2...",
                        "drafting_note": "Explanation of this subclause..."
                    }}
                ],
                "alternatives": [
                    {{
                        "alternative_text": "Alternative wording for the main clause...",
                        "drafting_note": "Detailed explanation of this alternative...",
                        "use_when": "Specific scenario when this alternative should be used..."
                    }}
                ]
            }},
            {{
                "type": "Scope of Work (Statement of Work) Clause",
                "clause_text": "Full text of clause 2...",
                "drafting_note": "Detailed explanation..."
    }}
  ]
}}

    PREDEFINED CLAUSE TYPES (type field MUST be exactly one of these values):
    - Fundamental Clauses: "Payment Terms Clause", "Scope of Work (Statement of Work) Clause", "Term & Termination Clause", "Price Adjustment / Escalation Clause"
    - Protective Clauses: "Indemnity Clause", "Limitation of Liability Clause", "Exemption / Exclusion Clause", "Liquidated Damages Clause", "Exculpatory Clause", "Gross-Up Clause", "Retention of Title (Romalpa) Clause"
    - Dispute Resolution Clauses: "Arbitration Clause", "Dispute Resolution / Escalation Clause", "Choice of Law Clause", "Confession of Judgment Clause"
    - Confidentiality & IP Clauses: "Confidentiality / Non-Disclosure Clause", "Intellectual Property Clause", "Non-Compete Clause", "Non-Solicitation Clause"
    - Operational (Boilerplate) Clauses: "Assignment Clause", "Change Control / Changes Clause", "Amendment Clause", "Notice Clause", "Severability Clause", "Survival Clause", "Entire Agreement Clause", "Waiver Clause", "Interpretation Clause", "Electronic Signatures Clause"

    IMPORTANT RULES:
    1. Inside any field that contains text from the contract (like "clause_text", "drafting_note"), you MUST escape any double quotes (") with a backslash (\\"). For example, if the text is 'The term "Agreement"...', it must be represented in the JSON as '"clause_text": "The term \\"Agreement\\"..."'.
    2. The "type" field must match exactly one of the predefined clause types listed above
    3. "subclauses" and "alternatives" are optional fields
    4. Ensure proper nesting and comma placement
    5. Remove any trailing commas
    6. Do not add any explanations or comments, just return the fixed JSON

    JSON Error Found: {error_message}

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

        
    def update_template_new_files(self, current_template: str, new_contract: str) -> str:
        """Update existing template with new contract content while preserving existing structure"""
        self.logger.info("Updating existing template with new contract content")
        
        if not self.api_config.is_configured():
            self.logger.error("API not configured")
            raise ValueError("API not configured")
        
        prompt = f"""You are an experienced solicitor in Hong Kong with decades of experience. You are the senior partner of an international law firm. 

    CONTEXT:
    You have an existing template that has already been refined through extensive iterations. This template contains:
    - Well-crafted legal language
    - Comprehensive drafting notes
    - Alternative clauses for different scenarios
    - Proper formatting and structure
    - Detailed guidance for junior lawyers

    TASK:
    Update this existing template by intelligently incorporating content from a new legal document. The goal is to enhance the template without disrupting its existing quality and structure.

    CRITICAL REQUIREMENTS:

    1. PRESERVE AND ENHANCE EXISTING CONTENT:
    - ENHANCE existing drafting notes with new insights from the new contract
    - UPDATE drafting notes to reflect any improvements or additional considerations
    - Maintain ALL existing alternative clauses and their explanations (but can improve them)
    - Preserve the existing template structure and organization
    - Keep all existing placeholder formats [Placeholder Name]
    - Keep all existing [DRAFTING NOTE: ...] and [ALTERNATIVE CLAUSE: ...] formatting
    - Maintain existing legal language and terminology (but can improve it)

    2. INTELLIGENT INTEGRATION:
    - Only add new content where it genuinely enhances the template
    - If a clause in the new contract is similar to an existing clause, improve the existing clause rather than replacing it
    - Add new clauses only if they bring significant value not already covered
    - If the new contract has better language for an existing concept, enhance the existing clause
    - Add new alternative clauses if the new contract provides genuinely different approaches
    - UPDATE existing drafting notes with new insights, enhanced explanations, and additional considerations from the new contract

    3. ENHANCEMENT APPROACH:
    - Enhance existing clauses with better language from the new contract
    - Add new placeholders if the new contract reveals additional variable fields
    - UPDATE and IMPROVE existing drafting notes with new insights and considerations
    - Add enhanced explanations to existing drafting notes based on new contract learnings
    - Add new alternative clauses only if they represent genuinely different legal approaches
    - Strengthen existing provisions with additional protections or clarity from the new contract
    - For each [DRAFTING NOTE: ...], review and enhance with new insights from the new contract where relevant
    - For each [ALTERNATIVE CLAUSE: ...], review and enhance explanations with new insights from the new contract where relevant

    4. FORMATTING CONSISTENCY:
    - Maintain all existing PREFIX formats for numbered lists:
        * NUMBERED_LIST_ITEM: [content]
        * SUB_NUMBERED_ITEM: [content]
        * SUB_SUB_NUMBERED_ITEM: [content]
        * ALPHA_ITEM_MAIN: [content]
        * SUB_ALPHA_ITEM: [content]
        * And all other existing formats
    - Preserve existing hierarchical structure
    - Keep existing cross-references and internal consistency

    5. WHAT NOT TO DO:
    - Do NOT remove existing drafting notes (but you CAN enhance and update them)
    - Do NOT remove existing alternative clauses (but you CAN improve them)
    - Do NOT change existing formatting or structure unnecessarily
    - Do NOT change existing [DRAFTING NOTE: ...] or [ALTERNATIVE CLAUSE: ...] formatting
    - Do NOT replace well-crafted existing clauses unless the new version is significantly better
    - Do NOT add redundant content that duplicates existing provisions

    6. DRAFTING NOTES ENHANCEMENT:
    - UPDATE existing drafting notes with new insights from the new contract
    - ADD additional considerations and best practices discovered in the new contract
    - ENHANCE existing guidance with more comprehensive explanations
    - UPDATE existing [DRAFTING NOTE: ...] sections with new insights and considerations
    - IMPROVE risk assessments and compliance considerations based on new contract
    - EXPAND "when to use" and "when not to use" guidance where the new contract provides insights
    - STRENGTHEN existing alternative clause explanations with additional context
    - UPDATE existing [ALTERNATIVE CLAUSE: ...], review and enhance explanations with new insights

    7. INTEGRATION STRATEGY:
    - Review each section of the new contract against the existing template
    - Identify genuine improvements or new concepts
    - Enhance existing clauses with better language where appropriate
    - Add new clauses only where they fill genuine gaps
    - Ensure any additions maintain the existing template's quality and style

    7. OUTPUT REQUIREMENTS:
    - Return the complete updated template
    - Maintain all existing quality and comprehensiveness
    - Ensure the template remains cohesive and well-organized
    - Preserve all existing guidance for junior lawyers
    - Keep all existing professional formatting and presentation

    EXISTING TEMPLATE:
    {current_template}

    NEW CONTRACT FOR INTEGRATION:
    {new_contract}

    Please update the template by intelligently incorporating the best elements from the new contract while preserving all existing quality, structure, and guidance. Pay special attention to updating and enhancing existing drafting notes with new insights, considerations, and improved explanations based on what you learn from the new contract."""
        
        try:
            self.logger.debug("Sending request to Gemini for intelligent template update")
            updated_template = self._call_gemini(prompt, temperature=0.1)
            
            self.logger.info("Successfully updated existing template")
            return updated_template
            
        except Exception as e:
            self.logger.error(f"Failed to update template: {str(e)}", exc_info=True)
            raise