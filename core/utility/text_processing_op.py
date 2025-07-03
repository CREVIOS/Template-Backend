"""
Text processing operation utilities for the Legal Template Generator.
Centralizes text analysis, content processing, formatting, and extraction operations.
"""

import re
import hashlib
import string
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from collections import Counter
from loguru import logger

logger.add("logs/text_processing_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# TEXT ANALYSIS AND METRICS
# ============================================================================

def analyze_text_content(text: str) -> Dict[str, Any]:
    """Comprehensive text analysis with various metrics"""
    try:
        if not text:
            return {
                "character_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "avg_word_length": 0,
                "avg_sentence_length": 0,
                "readability_score": 0,
                "complexity_indicators": {}
            }
        
        # Basic counts
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Averages
        avg_word_length = sum(len(word.strip(string.punctuation)) for word in text.split()) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Complexity indicators
        complexity = analyze_text_complexity(text)
        
        # Simple readability score (Flesch-like approximation)
        readability_score = calculate_readability_score(text, avg_sentence_length, avg_word_length)
        
        analysis = {
            "character_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "readability_score": round(readability_score, 2),
            "complexity_indicators": complexity
        }
        
        logger.debug(f"Text analysis completed: {word_count} words, {sentence_count} sentences")
        return analysis
        
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        return {"error": str(e)}

def analyze_text_complexity(text: str) -> Dict[str, Any]:
    """Analyze text complexity indicators"""
    try:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Count long words (>6 characters)
        long_words = [word for word in words if len(word.strip(string.punctuation)) > 6]
        long_word_percentage = (len(long_words) / len(words)) * 100 if words else 0
        
        # Count complex sentences (>20 words)
        complex_sentences = [sent for sent in sentences if len(sent.split()) > 20]
        complex_sentence_percentage = (len(complex_sentences) / len(sentences)) * 100 if sentences else 0
        
        # Legal terminology indicators
        legal_terms = count_legal_terminology(text)
        
        # Passive voice indicators
        passive_voice_count = count_passive_voice(text)
        
        complexity = {
            "long_words_count": len(long_words),
            "long_words_percentage": round(long_word_percentage, 2),
            "complex_sentences_count": len(complex_sentences),
            "complex_sentences_percentage": round(complex_sentence_percentage, 2),
            "legal_terms_count": legal_terms,
            "passive_voice_count": passive_voice_count,
            "avg_syllables_per_word": estimate_avg_syllables(text)
        }
        
        return complexity
        
    except Exception as e:
        logger.error(f"Text complexity analysis failed: {e}")
        return {}

def calculate_readability_score(text: str, avg_sentence_length: float, avg_word_length: float) -> float:
    """Calculate a simplified readability score"""
    try:
        # Simplified Flesch-like formula
        # Lower scores indicate more difficult text
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        return max(0, min(100, score))  # Clamp between 0-100
        
    except Exception as e:
        logger.error(f"Readability score calculation failed: {e}")
        return 0

def estimate_avg_syllables(text: str) -> float:
    """Estimate average syllables per word"""
    try:
        words = [word.strip(string.punctuation).lower() for word in text.split() if word.strip()]
        if not words:
            return 0
        
        total_syllables = sum(estimate_syllables(word) for word in words)
        return round(total_syllables / len(words), 2)
        
    except Exception as e:
        logger.error(f"Syllable estimation failed: {e}")
        return 0

def estimate_syllables(word: str) -> int:
    """Estimate syllables in a word"""
    try:
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        # Count vowel groups
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)
        
    except Exception as e:
        logger.error(f"Syllable estimation failed for word '{word}': {e}")
        return 1

# ============================================================================
# TEXT CLEANING AND FORMATTING
# ============================================================================

def clean_text_content(text: str, aggressive: bool = False) -> str:
    """Clean and normalize text content"""
    try:
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove or normalize special characters
        if aggressive:
            # More aggressive cleaning for analysis
            cleaned = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]', '', cleaned)
        
        # Normalize line endings
        cleaned = re.sub(r'\r\n', '\n', cleaned)
        cleaned = re.sub(r'\r', '\n', cleaned)
        
        # Remove excessive blank lines
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        
        logger.debug(f"Text cleaned: {len(text)} -> {len(cleaned)} characters")
        return cleaned
        
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text

def normalize_legal_text(text: str) -> str:
    """Normalize legal text with specific formatting rules"""
    try:
        # Clean base text
        normalized = clean_text_content(text)
        
        # Normalize legal citations
        normalized = normalize_legal_citations(normalized)
        
        # Normalize clause numbering
        normalized = normalize_clause_numbering(normalized)
        
        # Normalize legal terminology
        normalized = normalize_legal_terminology(normalized)
        
        # Normalize section references
        normalized = normalize_section_references(normalized)
        
        logger.debug("Legal text normalization completed")
        return normalized
        
    except Exception as e:
        logger.error(f"Legal text normalization failed: {e}")
        return text

def normalize_legal_citations(text: str) -> str:
    """Normalize legal citations to standard format"""
    try:
        # This is a simplified version - real legal citation normalization is complex
        
        # Normalize year citations
        text = re.sub(r'\(\s*(\d{4})\s*\)', r'(\1)', text)
        
        # Normalize section references
        text = re.sub(r'[Ss]ection\s+(\d+)', r'Section \1', text)
        text = re.sub(r'[Ss]ec\.\s*(\d+)', r'Section \1', text)
        
        # Normalize article references
        text = re.sub(r'[Aa]rticle\s+(\d+)', r'Article \1', text)
        text = re.sub(r'[Aa]rt\.\s*(\d+)', r'Article \1', text)
        
        return text
        
    except Exception as e:
        logger.error(f"Legal citation normalization failed: {e}")
        return text

def normalize_clause_numbering(text: str) -> str:
    """Normalize clause numbering formats"""
    try:
        # Convert various numbering formats to standard format
        
        # Handle Roman numerals
        text = re.sub(r'\b([IVX]+)\.\s*', r'\1. ', text)
        
        # Handle alphabetical numbering
        text = re.sub(r'\b([a-z])\)\s*', r'(\1) ', text)
        text = re.sub(r'\b([A-Z])\)\s*', r'(\1) ', text)
        
        # Handle numerical numbering
        text = re.sub(r'\b(\d+)\.\s*', r'\1. ', text)
        text = re.sub(r'\b(\d+)\)\s*', r'(\1) ', text)
        
        return text
        
    except Exception as e:
        logger.error(f"Clause numbering normalization failed: {e}")
        return text

def normalize_legal_terminology(text: str) -> str:
    """Normalize legal terminology to standard forms"""
    try:
        # Common legal term normalizations
        normalizations = {
            r'\bshall\b': 'shall',
            r'\bmay\b': 'may',
            r'\bnotwithstanding\b': 'notwithstanding',
            r'\bwhereas\b': 'whereas',
            r'\bwhereof\b': 'whereof',
            r'\bherein\b': 'herein',
            r'\bhereof\b': 'hereof',
            r'\bhereto\b': 'hereto',
            r'\bhereby\b': 'hereby',
            r'\bfurther\b': 'further',
            r'\bprovided\s+that\b': 'provided that',
            r'\bsubject\s+to\b': 'subject to',
            r'\bin\s+accordance\s+with\b': 'in accordance with',
            r'\bwith\s+respect\s+to\b': 'with respect to'
        }
        
        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
        
    except Exception as e:
        logger.error(f"Legal terminology normalization failed: {e}")
        return text

def normalize_section_references(text: str) -> str:
    """Normalize section and subsection references"""
    try:
        # Normalize section references
        text = re.sub(r'[Ss]ection\s+(\d+)\.(\d+)', r'Section \1.\2', text)
        text = re.sub(r'[Ss]ection\s+(\d+)\.(\d+)\.(\d+)', r'Section \1.\2.\3', text)
        
        # Normalize subsection references
        text = re.sub(r'[Ss]ubsection\s+\(([a-z])\)', r'subsection (\1)', text)
        text = re.sub(r'[Ss]ubsection\s+\((\d+)\)', r'subsection (\1)', text)
        
        return text
        
    except Exception as e:
        logger.error(f"Section reference normalization failed: {e}")
        return text

# ============================================================================
# TEXT EXTRACTION AND PARSING
# ============================================================================

def extract_placeholders(text: str) -> List[str]:
    """Extract placeholder fields from text (e.g., [Field Name])"""
    try:
        placeholders = re.findall(r'\[([^\]]+)\]', text)
        
        # Remove duplicates while preserving order
        unique_placeholders = []
        seen = set()
        for placeholder in placeholders:
            if placeholder not in seen:
                unique_placeholders.append(placeholder)
                seen.add(placeholder)
        
        logger.debug(f"Extracted {len(unique_placeholders)} unique placeholders")
        return unique_placeholders
        
    except Exception as e:
        logger.error(f"Placeholder extraction failed: {e}")
        return []

def extract_legal_entities(text: str) -> List[str]:
    """Extract legal entity names and types from text"""
    try:
        entities = []
        
        # Common legal entity patterns
        patterns = [
            r'\b([A-Z][a-zA-Z\s&]+(?:Limited|Ltd|Corporation|Corp|Inc|LLC|LLP|Partnership|Company|Co))\b',
            r'\b([A-Z][a-zA-Z\s&]+(?:Pte\.?\s*Ltd\.?|Sdn\.?\s*Bhd\.?))\b',
            r'\b([A-Z][a-zA-Z\s&]+(?:AG|GmbH|SA|SAS|BV|NV))\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # Remove duplicates
        entities = list(set(entities))
        
        logger.debug(f"Extracted {len(entities)} legal entities")
        return entities
        
    except Exception as e:
        logger.error(f"Legal entity extraction failed: {e}")
        return []

def extract_dates(text: str) -> List[str]:
    """Extract dates from text in various formats"""
    try:
        dates = []
        
        # Date patterns
        patterns = [
            r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
            r'\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b',  # YYYY/MM/DD
            r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        # Remove duplicates
        dates = list(set(dates))
        
        logger.debug(f"Extracted {len(dates)} dates")
        return dates
        
    except Exception as e:
        logger.error(f"Date extraction failed: {e}")
        return []

def extract_monetary_amounts(text: str) -> List[str]:
    """Extract monetary amounts from text"""
    try:
        amounts = []
        
        # Monetary amount patterns
        patterns = [
            r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|GBP|EUR|HKD|SGD|AUD|CAD)',  # 1,000.00 USD
            r'(?:USD|GBP|EUR|HKD|SGD|AUD|CAD)\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # USD 1,000.00
            r'(?:US\$|HK\$|S\$|A\$|C\$|£|€)\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?'  # US$1,000.00
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        
        # Remove duplicates
        amounts = list(set(amounts))
        
        logger.debug(f"Extracted {len(amounts)} monetary amounts")
        return amounts
        
    except Exception as e:
        logger.error(f"Monetary amount extraction failed: {e}")
        return []

def extract_email_addresses(text: str) -> List[str]:
    """Extract email addresses from text"""
    try:
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(pattern, text)
        
        # Remove duplicates
        emails = list(set(emails))
        
        logger.debug(f"Extracted {len(emails)} email addresses")
        return emails
        
    except Exception as e:
        logger.error(f"Email extraction failed: {e}")
        return []

def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text"""
    try:
        patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International format
            r'\(\d{3}\)\s?\d{3}-?\d{4}',  # US format (123) 456-7890
            r'\d{3}-\d{3}-\d{4}',  # US format 123-456-7890
            r'\d{3}\.\d{3}\.\d{4}',  # US format 123.456.7890
            r'\d{10,15}'  # Simple long number
        ]
        
        phones = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        # Filter out numbers that are too short or too long
        phones = [phone for phone in phones if 10 <= len(re.sub(r'\D', '', phone)) <= 15]
        
        # Remove duplicates
        phones = list(set(phones))
        
        logger.debug(f"Extracted {len(phones)} phone numbers")
        return phones
        
    except Exception as e:
        logger.error(f"Phone number extraction failed: {e}")
        return []

# ============================================================================
# TEXT SIMILARITY AND COMPARISON
# ============================================================================

def calculate_text_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """Calculate similarity between two texts using different methods"""
    try:
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = clean_text_content(text1.lower())
        text2 = clean_text_content(text2.lower())
        
        if method == "jaccard":
            return calculate_jaccard_similarity(text1, text2)
        elif method == "cosine":
            return calculate_cosine_similarity(text1, text2)
        elif method == "levenshtein":
            return calculate_levenshtein_similarity(text1, text2)
        else:
            logger.warning(f"Unknown similarity method: {method}, using jaccard")
            return calculate_jaccard_similarity(text1, text2)
        
    except Exception as e:
        logger.error(f"Text similarity calculation failed: {e}")
        return 0.0

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    try:
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Jaccard similarity calculation failed: {e}")
        return 0.0

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts"""
    try:
        # Create word frequency vectors
        words1 = text1.split()
        words2 = text2.split()
        
        all_words = set(words1 + words2)
        
        vector1 = [words1.count(word) for word in all_words]
        vector2 = [words2.count(word) for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = sum(a * a for a in vector1) ** 0.5
        magnitude2 = sum(b * b for b in vector2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
        
    except Exception as e:
        logger.error(f"Cosine similarity calculation failed: {e}")
        return 0.0

def calculate_levenshtein_similarity(text1: str, text2: str) -> float:
    """Calculate Levenshtein similarity between two texts"""
    try:
        distance = levenshtein_distance(text1, text2)
        max_length = max(len(text1), len(text2))
        
        return 1.0 - (distance / max_length) if max_length > 0 else 1.0
        
    except Exception as e:
        logger.error(f"Levenshtein similarity calculation failed: {e}")
        return 0.0

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    try:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
        
    except Exception as e:
        logger.error(f"Levenshtein distance calculation failed: {e}")
        return len(s1) + len(s2)

# ============================================================================
# TEXT HASHING AND FINGERPRINTING
# ============================================================================

def generate_text_hash(text: str, algorithm: str = "sha256") -> str:
    """Generate hash for text content"""
    try:
        # Normalize text for consistent hashing
        normalized = clean_text_content(text.lower())
        
        if algorithm == "md5":
            return hashlib.md5(normalized.encode('utf-8')).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(normalized.encode('utf-8')).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        else:
            logger.warning(f"Unknown hash algorithm: {algorithm}, using sha256")
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        
    except Exception as e:
        logger.error(f"Text hashing failed: {e}")
        return ""

def generate_text_fingerprint(text: str) -> str:
    """Generate a fingerprint for text content based on structure"""
    try:
        # Extract structural elements
        structure_elements = []
        
        # Word count bins
        word_count = len(text.split())
        structure_elements.append(f"words:{word_count//100}")
        
        # Character count bins
        char_count = len(text)
        structure_elements.append(f"chars:{char_count//1000}")
        
        # Common word patterns
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        word_frequencies = []
        for word in common_words:
            count = len(re.findall(r'\b' + word + r'\b', text.lower()))
            word_frequencies.append(f"{word}:{count}")
        
        structure_elements.extend(word_frequencies)
        
        # Generate fingerprint
        fingerprint_string = '|'.join(structure_elements)
        return hashlib.md5(fingerprint_string.encode('utf-8')).hexdigest()[:16]
        
    except Exception as e:
        logger.error(f"Text fingerprint generation failed: {e}")
        return ""

# ============================================================================
# SPECIALIZED LEGAL TEXT PROCESSING
# ============================================================================

def count_legal_terminology(text: str) -> int:
    """Count occurrences of legal terminology in text"""
    try:
        legal_terms = [
            'shall', 'may', 'agreement', 'contract', 'party', 'parties',
            'whereas', 'therefore', 'notwithstanding', 'pursuant', 'herein',
            'hereof', 'hereby', 'hereto', 'therein', 'thereof', 'thereby',
            'whereof', 'provided', 'subject to', 'in accordance with',
            'with respect to', 'indemnify', 'covenant', 'warranty',
            'representation', 'breach', 'default', 'termination',
            'jurisdiction', 'governing law', 'dispute resolution',
            'force majeure', 'liquidated damages', 'specific performance'
        ]
        
        count = 0
        text_lower = text.lower()
        
        for term in legal_terms:
            count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
        
        return count
        
    except Exception as e:
        logger.error(f"Legal terminology counting failed: {e}")
        return 0

def count_passive_voice(text: str) -> int:
    """Count passive voice constructions in text"""
    try:
        # Simple passive voice patterns
        passive_patterns = [
            r'\b(?:is|are|was|were|being|been|be)\s+\w+ed\b',
            r'\b(?:is|are|was|were|being|been|be)\s+\w+en\b',
            r'\b(?:is|are|was|were|being|been|be)\s+(?:given|taken|made|done|seen|known|shown|told|asked|brought|sent|found|left|kept|put|held|heard|felt|thought|said|come|gone)\b'
        ]
        
        count = 0
        for pattern in passive_patterns:
            matches = re.findall(pattern, text.lower())
            count += len(matches)
        
        return count
        
    except Exception as e:
        logger.error(f"Passive voice counting failed: {e}")
        return 0

def extract_clause_headers(text: str) -> List[Dict[str, Any]]:
    """Extract clause headers and their hierarchy from legal text"""
    try:
        headers = []
        
        # Common clause header patterns
        patterns = [
            (r'^(\d+)\.\s+(.+)$', 1),  # 1. Main Clause
            (r'^(\d+)\.(\d+)\s+(.+)$', 2),  # 1.1 Subclause
            (r'^(\d+)\.(\d+)\.(\d+)\s+(.+)$', 3),  # 1.1.1 Sub-subclause
            (r'^([A-Z]+)\.\s+(.+)$', 1),  # A. Main Clause
            (r'^([A-Z]+)\.(\d+)\s+(.+)$', 2),  # A.1 Subclause
            (r'^\(([a-z])\)\s+(.+)$', 2),  # (a) Subclause
            (r'^\(([ivx]+)\)\s+(.+)$', 2),  # (i) Roman numeral subclause
        ]
        
        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            for pattern, level in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    header_info = {
                        'line_number': line_num,
                        'level': level,
                        'full_text': line,
                        'number': match.group(1),
                        'title': match.groups()[-1].strip()
                    }
                    headers.append(header_info)
                    break
        
        logger.debug(f"Extracted {len(headers)} clause headers")
        return headers
        
    except Exception as e:
        logger.error(f"Clause header extraction failed: {e}")
        return []

def validate_clause_structure(text: str) -> Dict[str, Any]:
    """Validate the structure of clause numbering in legal text"""
    try:
        headers = extract_clause_headers(text)
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {
                'total_clauses': len(headers),
                'max_level': max([h['level'] for h in headers]) if headers else 0,
                'levels_distribution': {}
            }
        }
        
        # Check for sequential numbering
        level_counters = {}
        for header in headers:
            level = header['level']
            number = header['number']
            
            if level not in level_counters:
                level_counters[level] = []
            level_counters[level].append(number)
        
        # Validate sequential numbering for each level
        for level, numbers in level_counters.items():
            validation_results['statistics']['levels_distribution'][level] = len(numbers)
            
            # Check if numbers are sequential (for numeric clauses)
            if numbers and numbers[0].isdigit():
                numeric_numbers = [int(n) for n in numbers if n.isdigit()]
                if numeric_numbers:
                    expected = list(range(1, len(numeric_numbers) + 1))
                    if numeric_numbers != expected:
                        validation_results['errors'].append(
                            f"Non-sequential numbering at level {level}: expected {expected}, got {numeric_numbers}"
                        )
                        validation_results['is_valid'] = False
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Clause structure validation failed: {e}")
        return {'is_valid': False, 'errors': [str(e)]}

# ============================================================================
# TEXT TRUNCATION AND CHUNKING
# ============================================================================

def truncate_text(text: str, max_length: int, preserve_words: bool = True, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    try:
        if len(text) <= max_length:
            return text
        
        if preserve_words:
            # Find the last space before the max length
            truncated = text[:max_length - len(suffix)]
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]
            return truncated + suffix
        else:
            return text[:max_length - len(suffix)] + suffix
        
    except Exception as e:
        logger.error(f"Text truncation failed: {e}")
        return text[:max_length] if len(text) > max_length else text

def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into chunks with optional overlap"""
    try:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundaries
            if end < len(text):
                space_index = text.rfind(' ', start, end)
                if space_index > start:
                    end = space_index
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if overlap > 0 else end
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Text chunking failed: {e}")
        return [text]

# ============================================================================
# TEXT STATISTICS AND WORD FREQUENCY
# ============================================================================

def get_word_frequency(text: str, top_n: int = 20) -> List[Tuple[str, int]]:
    """Get word frequency statistics"""
    try:
        # Clean and normalize text
        cleaned = clean_text_content(text.lower())
        
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', cleaned)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'a', 'an',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Filter words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Return top N words
        return word_counts.most_common(top_n)
        
    except Exception as e:
        logger.error(f"Word frequency analysis failed: {e}")
        return []

def get_character_frequency(text: str) -> Dict[str, int]:
    """Get character frequency statistics"""
    try:
        char_counts = Counter(text.lower())
        return dict(char_counts)
        
    except Exception as e:
        logger.error(f"Character frequency analysis failed: {e}")
        return {} 