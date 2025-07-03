"""
Prompt management operation utilities for the Legal Template Generator.
Centralizes prompt templating, building, and optimization for AI interactions.
"""

import re
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from loguru import logger

logger.add("logs/prompt_management_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

class PromptTemplate:
    """A template for generating prompts with variable substitution"""
    
    def __init__(self, template: str, name: str = ""):
        self.template = template
        self.name = name
        self.variables = self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract variable names from template"""
        return re.findall(r'\{([^}]+)\}', self.template)
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables"""
        try:
            # Check for missing variables
            missing_vars = [var for var in self.variables if var not in kwargs]
            if missing_vars:
                logger.warning(f"Missing variables in template '{self.name}': {missing_vars}")
            
            # Substitute variables
            rendered = self.template
            for key, value in kwargs.items():
                rendered = rendered.replace(f'{{{key}}}', str(value))
            
            return rendered
            
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            return self.template

# ============================================================================
# LEGAL TEMPLATE GENERATION PROMPTS
# ============================================================================

INITIAL_TEMPLATE_GENERATION_PROMPT = PromptTemplate(
    template="""You are an experienced solicitor in Hong Kong. You have decades of experience and are the senior partner of an international law firm. Remember law is a practice of precision. Stick to the precedent's language and use of words as much as possible. If the clause is drafted long, then keep it long. If it's drafted short, keep it short. 

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
{contract_text}""",
    name="initial_template_generation"
)

TEMPLATE_UPDATE_PROMPT = PromptTemplate(
    template="""You are an experienced solicitor in Hong Kong. You have decades of experience and are the senior partner of an international law firm. Here is a second legal document that you think is well drafted. You are trying to make a template using legal documents so you can reuse it in the future easily. You initially generated a template using a legal document you recently drafted.
        Below you are given both the current template and also a New contract document. Update your template based on this second document. 

REQUIREMENTS:
1. Merge the best practices from both documents
2. Add new clauses to the template using the second document where appropriate
3. Improve existing clauses
4. Ensure ALL placeholders use square brackets [Placeholder Name]
5. Identify and add any new placeholder fields from this contract
6. Maintain comprehensive placeholder coverage for all variable information

{formatting_requirements}

{clause_requirements}

Current template:
{current_template}

New contract:
{new_contract}""",
    name="template_update"
)

DRAFTING_NOTES_PROMPT = PromptTemplate(
    template="""Using your legal documents, You have created a template for your junior lawyers. Now add comprehensive drafting notes to the template so that your junior lawyers can know when and how to use the template effectively. 
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

{formatting_requirements}

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
{template_text}""",
    name="drafting_notes"
)

METADATA_EXTRACTION_PROMPT = PromptTemplate(
    template="""Suppose, You are an experienced solicitor in Hong Kong. You have decades of experience and are the senior partner of an international law firm.
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

{contract_text}""",
    name="metadata_extraction"
)

CLAUSE_EXTRACTION_PROMPT = PromptTemplate(
    template="""You are an experienced legal analyst with expertise in contract clause identification and classification.
        
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
{template_content}""",
    name="clause_extraction"
)

JSON_REPAIR_PROMPT = PromptTemplate(
    template="""You are a JSON repair expert. The following text is a JSON response that has validation errors or is malformed.

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
{invalid_json}""",
    name="json_repair"
)

# ============================================================================
# PROMPT BUILDING FUNCTIONS
# ============================================================================

def build_initial_template_prompt(contract_text: str) -> str:
    """Build prompt for initial template generation"""
    try:
        return INITIAL_TEMPLATE_GENERATION_PROMPT.render(contract_text=contract_text)
    except Exception as e:
        logger.error(f"Failed to build initial template prompt: {e}")
        return f"Generate a legal template from the following contract:\n\n{contract_text}"

def build_template_update_prompt(current_template: str, new_contract: str) -> str:
    """Build prompt for template update"""
    try:
        formatting_requirements = get_formatting_requirements()
        clause_requirements = get_clause_requirements()
        
        return TEMPLATE_UPDATE_PROMPT.render(
            current_template=current_template,
            new_contract=new_contract,
            formatting_requirements=formatting_requirements,
            clause_requirements=clause_requirements
        )
    except Exception as e:
        logger.error(f"Failed to build template update prompt: {e}")
        return f"Update the template with new contract information:\n\nCurrent template:\n{current_template}\n\nNew contract:\n{new_contract}"

def build_drafting_notes_prompt(template_text: str, contracts_context: str) -> str:
    """Build prompt for adding drafting notes"""
    try:
        formatting_requirements = get_formatting_requirements()
        
        return DRAFTING_NOTES_PROMPT.render(
            template_text=template_text,
            contracts_context=contracts_context,
            formatting_requirements=formatting_requirements
        )
    except Exception as e:
        logger.error(f"Failed to build drafting notes prompt: {e}")
        return f"Add drafting notes to the template:\n\n{template_text}"

def build_metadata_extraction_prompt(contract_text: str, max_length: int = 8000) -> str:
    """Build prompt for metadata extraction"""
    try:
        # Truncate contract text if too long
        if len(contract_text) > max_length:
            contract_text = contract_text[:max_length]
            logger.warning(f"Contract text truncated to {max_length} characters for metadata extraction")
        
        return METADATA_EXTRACTION_PROMPT.render(contract_text=contract_text)
    except Exception as e:
        logger.error(f"Failed to build metadata extraction prompt: {e}")
        return f"Extract metadata from contract:\n\n{contract_text}"

def build_clause_extraction_prompt(template_content: str) -> str:
    """Build prompt for clause extraction"""
    try:
        return CLAUSE_EXTRACTION_PROMPT.render(template_content=template_content)
    except Exception as e:
        logger.error(f"Failed to build clause extraction prompt: {e}")
        return f"Extract clauses from template:\n\n{template_content}"

def build_json_repair_prompt(invalid_json: str, error_message: str) -> str:
    """Build prompt for JSON repair"""
    try:
        return JSON_REPAIR_PROMPT.render(
            invalid_json=invalid_json,
            error_message=error_message
        )
    except Exception as e:
        logger.error(f"Failed to build JSON repair prompt: {e}")
        return f"Fix this JSON:\n\n{invalid_json}"

# ============================================================================
# PROMPT COMPONENTS AND HELPERS
# ============================================================================

def get_formatting_requirements() -> str:
    """Get standard formatting requirements text"""
    return """7. CLAUSE LIST FORMATTING REQUIREMENTS FOR AUTOMATED PROCESSING AND WORD COMPATIBILITY:
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
   
   - USE THESE FORMATTINGS ONLY WHILE WRITING CLAUSE LISTS. WRITE OTHER PARAGRAPHS AND SECTIONS IN REGULAR MARKDOWN FORMAT."""

def get_clause_requirements() -> str:
    """Get standard clause requirements text"""
    return """8. CLAUSE STRUCTURE AND HIERARCHY REQUIREMENTS:
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
   - Maintain all signature blocks and execution provisions"""

def get_placeholder_guidelines() -> str:
    """Get placeholder usage guidelines"""
    return """PLACEHOLDER GUIDELINES:
1. Use square brackets for ALL placeholders: [Placeholder Name]
2. Common placeholder categories:
   - Party information: [Party Name], [Company Name], [Client Name]
   - Dates: [Effective Date], [Contract Date], [Termination Date]
   - Financial: [Contract Amount], [Payment Amount], [Currency]
   - Contact: [Email Address], [Phone Number], [Address]
   - Legal: [Jurisdiction], [Governing Law], [Registration Number]
   - Terms: [Payment Terms], [Notice Period], [Service Period]
3. Make placeholders descriptive and specific
4. Ensure all variable content is properly placeholder-ized
5. Use consistent naming conventions"""

# ============================================================================
# PROMPT OPTIMIZATION AND VALIDATION
# ============================================================================

def optimize_prompt_length(prompt: str, max_length: int = 100000) -> str:
    """Optimize prompt length while preserving important content"""
    try:
        if len(prompt) <= max_length:
            return prompt
        
        logger.warning(f"Prompt too long ({len(prompt)} chars), optimizing to {max_length}")
        
        # Split into sections
        sections = prompt.split('\n\n')
        
        # Keep essential sections (those with REQUIREMENTS, CRITICAL, etc.)
        essential_sections = []
        optional_sections = []
        
        for section in sections:
            section_upper = section.upper()
            if any(keyword in section_upper for keyword in ['REQUIREMENTS', 'CRITICAL', 'IMPORTANT', 'TASK']):
                essential_sections.append(section)
            else:
                optional_sections.append(section)
        
        # Start with essential sections
        optimized = '\n\n'.join(essential_sections)
        
        # Add optional sections if space allows
        for section in optional_sections:
            if len(optimized) + len(section) + 2 <= max_length:
                optimized += '\n\n' + section
            else:
                break
        
        # If still too long, truncate with warning
        if len(optimized) > max_length:
            optimized = optimized[:max_length - 100] + '\n\n[Content truncated due to length limits]'
        
        logger.info(f"Prompt optimized from {len(prompt)} to {len(optimized)} characters")
        return optimized
        
    except Exception as e:
        logger.error(f"Prompt optimization failed: {e}")
        return prompt[:max_length] if len(prompt) > max_length else prompt

def validate_prompt_structure(prompt: str) -> Dict[str, Any]:
    """Validate prompt structure and content"""
    try:
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'suggestions': [],
            'statistics': {
                'total_length': len(prompt),
                'word_count': len(prompt.split()),
                'placeholder_count': len(re.findall(r'\{[^}]+\}', prompt)),
                'section_count': len(prompt.split('\n\n'))
            }
        }
        
        # Check for common issues
        if len(prompt) > 100000:
            validation_results['warnings'].append("Prompt may be too long for some AI models")
        
        if '{' in prompt and '}' in prompt:
            unresolved_placeholders = re.findall(r'\{([^}]+)\}', prompt)
            if unresolved_placeholders:
                validation_results['warnings'].append(f"Unresolved placeholders: {unresolved_placeholders}")
        
        if not any(keyword in prompt.upper() for keyword in ['REQUIREMENTS', 'TASK', 'INSTRUCTIONS']):
            validation_results['suggestions'].append("Consider adding clear requirements or task instructions")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Prompt validation failed: {e}")
        return {'is_valid': False, 'error': str(e)}

# ============================================================================
# CUSTOM PROMPT BUILDERS
# ============================================================================

def build_custom_prompt(
    instruction: str,
    context: str = "",
    requirements: List[str] = None,
    examples: List[str] = None,
    output_format: str = ""
) -> str:
    """Build a custom prompt with specified components"""
    try:
        prompt_parts = []
        
        # Add instruction
        if instruction:
            prompt_parts.append(instruction)
        
        # Add context
        if context:
            prompt_parts.append(f"CONTEXT:\n{context}")
        
        # Add requirements
        if requirements:
            req_text = "\n".join(f"- {req}" for req in requirements)
            prompt_parts.append(f"REQUIREMENTS:\n{req_text}")
        
        # Add examples
        if examples:
            examples_text = "\n\n".join(f"Example {i+1}:\n{example}" for i, example in enumerate(examples))
            prompt_parts.append(f"EXAMPLES:\n{examples_text}")
        
        # Add output format
        if output_format:
            prompt_parts.append(f"OUTPUT FORMAT:\n{output_format}")
        
        return "\n\n".join(prompt_parts)
        
    except Exception as e:
        logger.error(f"Custom prompt building failed: {e}")
        return instruction

def build_comparison_prompt(item1: str, item2: str, comparison_type: str = "similarity") -> str:
    """Build prompt for comparing two items"""
    try:
        if comparison_type == "similarity":
            instruction = "Compare the following two items and identify their similarities and differences."
        elif comparison_type == "legal_analysis":
            instruction = "Analyze the following two legal documents and provide a comprehensive comparison."
        elif comparison_type == "improvement":
            instruction = "Compare these two versions and suggest improvements."
        else:
            instruction = f"Compare the following two items for {comparison_type}."
        
        return build_custom_prompt(
            instruction=instruction,
            context=f"Item 1:\n{item1}\n\nItem 2:\n{item2}",
            requirements=[
                "Provide detailed analysis",
                "Highlight key differences",
                "Suggest improvements where applicable"
            ]
        )
        
    except Exception as e:
        logger.error(f"Comparison prompt building failed: {e}")
        return f"Compare:\n\n{item1}\n\nWith:\n\n{item2}"

# ============================================================================
# PROMPT CACHING AND MANAGEMENT
# ============================================================================

class PromptCache:
    """Simple prompt caching system"""
    
    def __init__(self):
        self.cache = {}
        self.usage_stats = {}
    
    def get_prompt(self, prompt_name: str, **kwargs) -> Optional[str]:
        """Get cached prompt or None if not found"""
        try:
            cache_key = f"{prompt_name}:{hash(str(sorted(kwargs.items())))}"
            
            if cache_key in self.cache:
                self.usage_stats[cache_key] = self.usage_stats.get(cache_key, 0) + 1
                logger.debug(f"Retrieved cached prompt: {prompt_name}")
                return self.cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Prompt cache retrieval failed: {e}")
            return None
    
    def cache_prompt(self, prompt_name: str, prompt: str, **kwargs) -> None:
        """Cache a prompt"""
        try:
            cache_key = f"{prompt_name}:{hash(str(sorted(kwargs.items())))}"
            self.cache[cache_key] = prompt
            self.usage_stats[cache_key] = 0
            logger.debug(f"Cached prompt: {prompt_name}")
            
        except Exception as e:
            logger.error(f"Prompt caching failed: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached prompts"""
        self.cache.clear()
        self.usage_stats.clear()
        logger.info("Prompt cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_cached': len(self.cache),
            'total_usage': sum(self.usage_stats.values()),
            'most_used': max(self.usage_stats.items(), key=lambda x: x[1]) if self.usage_stats else None
        }

# Global prompt cache instance
prompt_cache = PromptCache()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_tokens_estimate(text: str, model: str = "gpt-3.5") -> int:
    """Rough estimate of token count for different models"""
    try:
        if model.startswith("gpt"):
            # GPT models: roughly 4 characters per token
            return len(text) // 4
        elif model.startswith("gemini"):
            # Gemini models: roughly 4 characters per token
            return len(text) // 4
        else:
            # Default: word count
            return len(text.split())
            
    except Exception as e:
        logger.error(f"Token estimation failed: {e}")
        return 0

def extract_prompt_variables(prompt: str) -> List[str]:
    """Extract all variables from a prompt template"""
    try:
        return re.findall(r'\{([^}]+)\}', prompt)
    except Exception as e:
        logger.error(f"Variable extraction failed: {e}")
        return [] 