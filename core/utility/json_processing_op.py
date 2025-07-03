"""
JSON processing operation utilities for the Legal Template Generator.
Centralizes JSON validation, parsing, repair, and transformation operations.
"""

import json
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from loguru import logger

logger.add("logs/json_processing_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# JSON VALIDATION AND PARSING
# ============================================================================

def validate_json_string(json_string: str) -> Dict[str, Any]:
    """Validate and parse a JSON string with comprehensive error handling"""
    try:
        if not json_string or not json_string.strip():
            logger.warning("Empty or whitespace-only JSON string provided")
            raise ValueError("Empty JSON string")
        
        # Clean the JSON string
        cleaned_json = clean_json_string(json_string)
        
        # Parse the JSON
        parsed_data = json.loads(cleaned_json)
        
        logger.debug(f"Successfully parsed JSON with {len(str(parsed_data))} characters")
        return parsed_data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"JSON validation failed: {e}")
        raise

def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate JSON data against a simple schema definition"""
    try:
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                raise ValueError(f"Missing required field: {field}")
        
        # Check field types
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                field_type = field_schema.get("type")
                field_value = data[field]
                
                if field_type == "string" and not isinstance(field_value, str):
                    raise ValueError(f"Field '{field}' must be a string, got {type(field_value).__name__}")
                elif field_type == "number" and not isinstance(field_value, (int, float)):
                    raise ValueError(f"Field '{field}' must be a number, got {type(field_value).__name__}")
                elif field_type == "integer" and not isinstance(field_value, int):
                    raise ValueError(f"Field '{field}' must be an integer, got {type(field_value).__name__}")
                elif field_type == "boolean" and not isinstance(field_value, bool):
                    raise ValueError(f"Field '{field}' must be a boolean, got {type(field_value).__name__}")
                elif field_type == "array" and not isinstance(field_value, list):
                    raise ValueError(f"Field '{field}' must be an array, got {type(field_value).__name__}")
                elif field_type == "object" and not isinstance(field_value, dict):
                    raise ValueError(f"Field '{field}' must be an object, got {type(field_value).__name__}")
        
        logger.debug("JSON schema validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise

def is_valid_json(json_string: str) -> bool:
    """Check if a string is valid JSON without raising exceptions"""
    try:
        json.loads(json_string)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

# ============================================================================
# JSON CLEANING AND REPAIR
# ============================================================================

def clean_json_string(json_string: str) -> str:
    """Clean common JSON formatting issues"""
    try:
        # Remove leading/trailing whitespace
        cleaned = json_string.strip()
        
        # Remove potential markdown code block markers
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        # Remove any leading/trailing quotes that might wrap the entire JSON
        cleaned = cleaned.strip()
        if cleaned.startswith('"') and cleaned.endswith('"') and cleaned.count('"') == 2:
            cleaned = cleaned[1:-1]
        
        # Fix common quote issues
        cleaned = fix_quote_issues(cleaned)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"JSON cleaning failed: {e}")
        return json_string

def fix_quote_issues(json_string: str) -> str:
    """Fix common quote-related issues in JSON strings"""
    try:
        # Fix unescaped quotes in string values
        # This is a simplified fix - more complex cases might need AI repair
        
        # Fix single quotes to double quotes (basic case)
        # Only replace single quotes that are clearly meant to be JSON quotes
        fixed = re.sub(r"'([^']*)':", r'"\1":', json_string)  # Keys
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)  # Values
        
        # Fix trailing commas
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        return fixed
        
    except Exception as e:
        logger.error(f"Quote fixing failed: {e}")
        return json_string

def repair_json_basic(json_string: str) -> Optional[str]:
    """Attempt basic JSON repair for common issues"""
    try:
        # First try cleaning
        cleaned = clean_json_string(json_string)
        
        # Try to parse - if successful, return
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass
        
        # Try fixing common issues
        repaired = cleaned
        
        # Fix missing commas
        repaired = re.sub(r'"\s*\n\s*"', '",\n    "', repaired)
        repaired = re.sub(r'}\s*\n\s*"', '},\n    "', repaired)
        repaired = re.sub(r']\s*\n\s*"', '],\n    "', repaired)
        
        # Fix missing closing brackets/braces
        open_braces = repaired.count('{')
        close_braces = repaired.count('}')
        open_brackets = repaired.count('[')
        close_brackets = repaired.count(']')
        
        if open_braces > close_braces:
            repaired += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            repaired += ']' * (open_brackets - close_brackets)
        
        # Try to parse again
        try:
            json.loads(repaired)
            logger.info("Successfully repaired JSON with basic repair")
            return repaired
        except json.JSONDecodeError:
            logger.warning("Basic JSON repair failed")
            return None
            
    except Exception as e:
        logger.error(f"Basic JSON repair failed: {e}")
        return None

# ============================================================================
# JSON TRANSFORMATION AND FORMATTING
# ============================================================================

def pretty_format_json(data: Union[Dict, List, str], indent: int = 2) -> str:
    """Format JSON data with proper indentation"""
    try:
        if isinstance(data, str):
            # Try to parse if it's a string
            data = json.loads(data)
        
        return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=True)
        
    except Exception as e:
        logger.error(f"JSON formatting failed: {e}")
        return str(data)

def compact_json(data: Union[Dict, List, str]) -> str:
    """Format JSON data as compact string"""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        
        return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"JSON compacting failed: {e}")
        return str(data)

def flatten_json(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Flatten nested JSON structure"""
    try:
        def _flatten(obj, parent_key=''):
            items = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{parent_key}{separator}{key}" if parent_key else key
                    if isinstance(value, dict):
                        items.extend(_flatten(value, new_key).items())
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            list_key = f"{new_key}{separator}{i}"
                            if isinstance(item, dict):
                                items.extend(_flatten(item, list_key).items())
                            else:
                                items.append((list_key, item))
                    else:
                        items.append((new_key, value))
            return dict(items)
        
        flattened = _flatten(data)
        logger.debug(f"Flattened JSON from {len(data)} to {len(flattened)} keys")
        return flattened
        
    except Exception as e:
        logger.error(f"JSON flattening failed: {e}")
        return data

def unflatten_json(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Unflatten a flattened JSON structure"""
    try:
        result = {}
        
        for key, value in data.items():
            keys = key.split(separator)
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    # Check if next key is numeric (for arrays)
                    next_key = keys[keys.index(k) + 1]
                    if next_key.isdigit():
                        current[k] = []
                    else:
                        current[k] = {}
                current = current[k]
            
            final_key = keys[-1]
            if final_key.isdigit():
                # Handle array indices
                index = int(final_key)
                if not isinstance(current, list):
                    current = []
                # Extend list if necessary
                while len(current) <= index:
                    current.append(None)
                current[index] = value
            else:
                current[final_key] = value
        
        logger.debug(f"Unflattened JSON to {len(result)} top-level keys")
        return result
        
    except Exception as e:
        logger.error(f"JSON unflattening failed: {e}")
        return data

# ============================================================================
# JSON COMPARISON AND DIFFING
# ============================================================================

def compare_json_objects(obj1: Dict[str, Any], obj2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two JSON objects and return differences"""
    try:
        differences = {
            "added": {},
            "removed": {},
            "modified": {},
            "unchanged": {}
        }
        
        all_keys = set(obj1.keys()) | set(obj2.keys())
        
        for key in all_keys:
            if key in obj1 and key in obj2:
                if obj1[key] == obj2[key]:
                    differences["unchanged"][key] = obj1[key]
                else:
                    differences["modified"][key] = {
                        "old": obj1[key],
                        "new": obj2[key]
                    }
            elif key in obj1:
                differences["removed"][key] = obj1[key]
            else:
                differences["added"][key] = obj2[key]
        
        logger.debug(f"JSON comparison completed: {len(differences['added'])} added, {len(differences['removed'])} removed, {len(differences['modified'])} modified")
        return differences
        
    except Exception as e:
        logger.error(f"JSON comparison failed: {e}")
        return {"error": str(e)}

def merge_json_objects(obj1: Dict[str, Any], obj2: Dict[str, Any], strategy: str = "override") -> Dict[str, Any]:
    """Merge two JSON objects with different strategies"""
    try:
        if strategy == "override":
            # obj2 overrides obj1
            result = obj1.copy()
            result.update(obj2)
            
        elif strategy == "merge":
            # Deep merge
            result = obj1.copy()
            for key, value in obj2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_json_objects(result[key], value, strategy)
                else:
                    result[key] = value
                    
        elif strategy == "keep_original":
            # Keep original values, only add new keys
            result = obj1.copy()
            for key, value in obj2.items():
                if key not in result:
                    result[key] = value
                    
        else:
            logger.warning(f"Unknown merge strategy: {strategy}, using override")
            result = obj1.copy()
            result.update(obj2)
        
        logger.debug(f"JSON merge completed using {strategy} strategy")
        return result
        
    except Exception as e:
        logger.error(f"JSON merge failed: {e}")
        return obj1

# ============================================================================
# JSON EXTRACTION AND FILTERING
# ============================================================================

def extract_json_values(data: Dict[str, Any], path: str) -> List[Any]:
    """Extract values from JSON using dot notation path"""
    try:
        keys = path.split('.')
        results = []
        
        def _extract(obj, keys_remaining):
            if not keys_remaining:
                results.append(obj)
                return
            
            key = keys_remaining[0]
            remaining = keys_remaining[1:]
            
            if isinstance(obj, dict):
                if key in obj:
                    _extract(obj[key], remaining)
                elif key == '*':
                    for value in obj.values():
                        _extract(value, remaining)
            elif isinstance(obj, list):
                if key.isdigit():
                    index = int(key)
                    if 0 <= index < len(obj):
                        _extract(obj[index], remaining)
                elif key == '*':
                    for item in obj:
                        _extract(item, remaining)
        
        _extract(data, keys)
        logger.debug(f"Extracted {len(results)} values for path '{path}'")
        return results
        
    except Exception as e:
        logger.error(f"JSON value extraction failed: {e}")
        return []

def filter_json_by_keys(data: Dict[str, Any], keys: List[str], include: bool = True) -> Dict[str, Any]:
    """Filter JSON object by including or excluding specific keys"""
    try:
        if include:
            # Include only specified keys
            result = {k: v for k, v in data.items() if k in keys}
        else:
            # Exclude specified keys
            result = {k: v for k, v in data.items() if k not in keys}
        
        logger.debug(f"Filtered JSON: {len(result)} keys remaining from {len(data)} original keys")
        return result
        
    except Exception as e:
        logger.error(f"JSON filtering failed: {e}")
        return data

def filter_json_by_values(data: Dict[str, Any], condition: callable) -> Dict[str, Any]:
    """Filter JSON object by values using a condition function"""
    try:
        result = {}
        
        for key, value in data.items():
            try:
                if condition(value):
                    result[key] = value
            except Exception as e:
                logger.warning(f"Condition evaluation failed for key '{key}': {e}")
        
        logger.debug(f"Value-based filtering: {len(result)} keys remaining from {len(data)} original keys")
        return result
        
    except Exception as e:
        logger.error(f"JSON value filtering failed: {e}")
        return data

# ============================================================================
# JSON CONVERSION AND SERIALIZATION
# ============================================================================

def json_to_csv_compatible(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert JSON data to CSV-compatible format (flatten nested objects)"""
    try:
        if not data or not isinstance(data, list):
            logger.error("Data must be a list of dictionaries")
            return []
        
        csv_data = []
        for item in data:
            if isinstance(item, dict):
                flattened = flatten_json(item)
                csv_data.append(flattened)
            else:
                csv_data.append({"value": item})
        
        logger.debug(f"Converted {len(data)} JSON objects to CSV-compatible format")
        return csv_data
        
    except Exception as e:
        logger.error(f"JSON to CSV conversion failed: {e}")
        return []

def safe_json_serialize(data: Any) -> str:
    """Safely serialize data to JSON with custom handling for non-serializable types"""
    try:
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        return json.dumps(data, default=default_serializer, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Safe JSON serialization failed: {e}")
        return json.dumps({"error": f"Serialization failed: {e}"})

# ============================================================================
# JSON STATISTICS AND ANALYSIS
# ============================================================================

def analyze_json_structure(data: Union[Dict, List]) -> Dict[str, Any]:
    """Analyze JSON structure and provide statistics"""
    try:
        stats = {
            "total_keys": 0,
            "depth": 0,
            "types": {},
            "null_values": 0,
            "empty_values": 0,
            "array_lengths": [],
            "size_bytes": 0
        }
        
        def _analyze(obj, depth=0):
            stats["depth"] = max(stats["depth"], depth)
            
            if isinstance(obj, dict):
                stats["total_keys"] += len(obj)
                for key, value in obj.items():
                    _analyze_value(value, depth + 1)
            elif isinstance(obj, list):
                stats["array_lengths"].append(len(obj))
                for item in obj:
                    _analyze_value(item, depth + 1)
            else:
                _analyze_value(obj, depth)
        
        def _analyze_value(value, depth):
            value_type = type(value).__name__
            stats["types"][value_type] = stats["types"].get(value_type, 0) + 1
            
            if value is None:
                stats["null_values"] += 1
            elif (isinstance(value, (str, list, dict)) and len(value) == 0):
                stats["empty_values"] += 1
            
            if isinstance(value, (dict, list)):
                _analyze(value, depth)
        
        _analyze(data)
        
        # Calculate size
        stats["size_bytes"] = len(json.dumps(data, ensure_ascii=False))
        
        # Calculate statistics
        if stats["array_lengths"]:
            stats["avg_array_length"] = sum(stats["array_lengths"]) / len(stats["array_lengths"])
            stats["max_array_length"] = max(stats["array_lengths"])
            stats["min_array_length"] = min(stats["array_lengths"])
        
        logger.debug(f"JSON analysis completed: {stats['total_keys']} keys, depth {stats['depth']}")
        return stats
        
    except Exception as e:
        logger.error(f"JSON analysis failed: {e}")
        return {"error": str(e)}

# ============================================================================
# JSON VALIDATION FOR SPECIFIC SCHEMAS
# ============================================================================

def validate_clause_json(data: Dict[str, Any]) -> bool:
    """Validate JSON data for clause structure"""
    try:
        required_fields = ["clause_type", "clause_text", "position_context", "clause_purpose"]
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required clause field: {field}")
                return False
            
            if not isinstance(data[field], str) or not data[field].strip():
                logger.error(f"Clause field '{field}' must be a non-empty string")
                return False
        
        # Validate relevance_assessment if present
        if "relevance_assessment" in data:
            relevance = data["relevance_assessment"]
            if not isinstance(relevance, dict):
                logger.error("relevance_assessment must be an object")
                return False
            
            expected_relevance_fields = [
                "when_to_include", "when_to_exclude", "industry_considerations",
                "risk_implications", "compliance_requirements", "best_practices"
            ]
            
            for field in expected_relevance_fields:
                if field in relevance and not isinstance(relevance[field], list):
                    logger.error(f"Relevance field '{field}' must be an array")
                    return False
        
        logger.debug("Clause JSON validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Clause JSON validation failed: {e}")
        return False

def validate_template_json(data: Dict[str, Any]) -> bool:
    """Validate JSON data for template structure"""
    try:
        required_fields = ["name", "content", "folder_id"]
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required template field: {field}")
                return False
        
        # Validate types
        if not isinstance(data["name"], str) or not data["name"].strip():
            logger.error("Template name must be a non-empty string")
            return False
        
        if not isinstance(data["content"], str):
            logger.error("Template content must be a string")
            return False
        
        if not isinstance(data["folder_id"], str) or not data["folder_id"].strip():
            logger.error("Template folder_id must be a non-empty string")
            return False
        
        logger.debug("Template JSON validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Template JSON validation failed: {e}")
        return False 