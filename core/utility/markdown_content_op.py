"""
Markdown content operation utilities for the Legal Template Generator.
Centralizes all markdown content database operations and text processing.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from datetime import datetime, timedelta
from loguru import logger
from core.database import get_database_service

logger.add("logs/markdown_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# MARKDOWN CONTENT CRUD OPERATIONS
# ============================================================================

async def save_markdown_content(
    file_id: str,
    user_id: str,
    content: str,
    word_count: Optional[int] = None,
    content_id: Optional[str] = None
) -> Optional[str]:
    """Save markdown content for a file"""
    try:
        db = get_database_service()
        
        # Calculate word count if not provided
        if word_count is None:
            word_count = len(content.split())
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        markdown_data = {
            "id": content_id or str(uuid4()),
            "file_id": file_id,
            "user_id": user_id,
            "content": content,
            "word_count": word_count,
            "content_hash": content_hash,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("markdown_content").insert(markdown_data).execute()
        
        if response.data:
            logger.info(f"Saved markdown content for file {file_id} ({word_count} words)")
            return markdown_data["id"]
        return None
        
    except Exception as e:
        logger.error(f"Error saving markdown content for file {file_id}: {e}")
        return None

async def get_markdown_content(file_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get markdown content for a file"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("markdown_content").select("*").eq("file_id", file_id).eq("user_id", user_id).single().execute()
        
        return response.data if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting markdown content for file {file_id}: {e}")
        return None

async def get_markdown_content_by_id(content_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get markdown content by ID"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("markdown_content").select("*").eq("id", content_id).eq("user_id", user_id).single().execute()
        
        return response.data if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting markdown content by ID {content_id}: {e}")
        return None

async def update_markdown_content(
    file_id: str,
    user_id: str,
    content: str,
    word_count: Optional[int] = None
) -> bool:
    """Update markdown content for a file"""
    try:
        db = get_database_service()
        
        # Calculate word count if not provided
        if word_count is None:
            word_count = len(content.split())
        
        # Generate content hash
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        update_data = {
            "content": content,
            "word_count": word_count,
            "content_hash": content_hash,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("markdown_content").update(update_data).eq("file_id", file_id).eq("user_id", user_id).execute()
        
        if response.data:
            logger.info(f"Updated markdown content for file {file_id} ({word_count} words)")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating markdown content for file {file_id}: {e}")
        return False

async def delete_markdown_content(file_id: str, user_id: str) -> bool:
    """Delete markdown content for a file"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("markdown_content").delete().eq("file_id", file_id).eq("user_id", user_id).execute()
        
        if response.data:
            logger.info(f"Deleted markdown content for file {file_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error deleting markdown content for file {file_id}: {e}")
        return False

# ============================================================================
# MARKDOWN CONTENT QUERIES AND FILTERING
# ============================================================================

async def get_markdown_contents_by_user(
    user_id: str,
    folder_id: Optional[str] = None,
    min_word_count: Optional[int] = None,
    max_word_count: Optional[int] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get markdown contents for a user with optional filters"""
    try:
        db = get_database_service()
        
        # Build query with file join
        query = db.client.from_("markdown_content").select(
            "*, files(original_filename, folder_id, status, created_at as file_created_at)"
        ).eq("user_id", user_id)
        
        # Apply filters
        if folder_id:
            query = query.eq("files.folder_id", folder_id)
        if min_word_count is not None:
            query = query.gte("word_count", min_word_count)
        if max_word_count is not None:
            query = query.lte("word_count", max_word_count)
            
        response = await query.order("created_at", desc=True).limit(limit).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting markdown contents for user {user_id}: {e}")
        return []

async def get_markdown_contents_by_folder(
    folder_id: str,
    user_id: str,
    include_content: bool = False
) -> List[Dict[str, Any]]:
    """Get markdown contents for a specific folder"""
    try:
        db = get_database_service()
        
        # Build select fields
        select_fields = "id, file_id, user_id, word_count, content_hash, created_at, updated_at"
        if include_content:
            select_fields = "*"
        
        # Join with files table to filter by folder
        query = db.client.from_("markdown_content").select(
            f"{select_fields}, files(original_filename, status)"
        ).eq("user_id", user_id).eq("files.folder_id", folder_id)
        
        response = await query.order("created_at", desc=True).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting markdown contents for folder {folder_id}: {e}")
        return []

async def search_markdown_content(
    user_id: str,
    search_term: str,
    folder_id: Optional[str] = None,
    min_word_count: Optional[int] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Search markdown content by text"""
    try:
        db = get_database_service()
        
        # Build query
        query = db.client.from_("markdown_content").select(
            "*, files(original_filename, folder_id, status)"
        ).eq("user_id", user_id).ilike("content", f"%{search_term}%")
        
        # Apply filters
        if folder_id:
            query = query.eq("files.folder_id", folder_id)
        if min_word_count is not None:
            query = query.gte("word_count", min_word_count)
            
        response = await query.order("updated_at", desc=True).limit(limit).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error searching markdown content for '{search_term}': {e}")
        return []

# ============================================================================
# MARKDOWN CONTENT ANALYSIS AND PROCESSING
# ============================================================================

def extract_headings(content: str) -> List[Dict[str, Any]]:
    """Extract headings from markdown content"""
    try:
        headings = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Match markdown headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                
                headings.append({
                    "level": level,
                    "text": text,
                    "line_number": line_num
                })
        
        return headings
        
    except Exception as e:
        logger.error(f"Error extracting headings: {e}")
        return []

def extract_links(content: str) -> List[Dict[str, Any]]:
    """Extract links from markdown content"""
    try:
        links = []
        
        # Match markdown links [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.finditer(link_pattern, content)
        
        for match in matches:
            links.append({
                "text": match.group(1),
                "url": match.group(2),
                "position": match.start()
            })
        
        return links
        
    except Exception as e:
        logger.error(f"Error extracting links: {e}")
        return []

def extract_code_blocks(content: str) -> List[Dict[str, Any]]:
    """Extract code blocks from markdown content"""
    try:
        code_blocks = []
        
        # Match fenced code blocks
        code_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.finditer(code_pattern, content, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2)
            
            code_blocks.append({
                "language": language,
                "code": code,
                "position": match.start(),
                "length": len(code)
            })
        
        return code_blocks
        
    except Exception as e:
        logger.error(f"Error extracting code blocks: {e}")
        return []

def calculate_reading_time(content: str, words_per_minute: int = 200) -> Dict[str, Any]:
    """Calculate estimated reading time for markdown content"""
    try:
        # Remove markdown formatting for accurate word count
        clean_content = re.sub(r'[#*_`\[\]()]+', '', content)
        words = len(clean_content.split())
        
        # Calculate reading time
        minutes = words / words_per_minute
        seconds = (minutes % 1) * 60
        
        return {
            "words": words,
            "minutes": int(minutes),
            "seconds": int(seconds),
            "total_seconds": int(minutes * 60 + seconds),
            "readable_time": f"{int(minutes)}m {int(seconds)}s" if minutes >= 1 else f"{int(seconds)}s"
        }
        
    except Exception as e:
        logger.error(f"Error calculating reading time: {e}")
        return {"words": 0, "minutes": 0, "seconds": 0, "total_seconds": 0, "readable_time": "0s"}

async def analyze_markdown_content(content_id: str, user_id: str) -> Dict[str, Any]:
    """Analyze markdown content and return comprehensive statistics"""
    try:
        # Get markdown content
        content_data = await get_markdown_content_by_id(content_id, user_id)
        if not content_data:
            return {}
        
        content = content_data.get("content", "")
        
        # Extract various elements
        headings = extract_headings(content)
        links = extract_links(content)
        code_blocks = extract_code_blocks(content)
        reading_time = calculate_reading_time(content)
        
        # Calculate additional statistics
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Character statistics
        char_count = len(content)
        char_count_no_spaces = len(content.replace(' ', ''))
        
        # Paragraph estimation
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        
        return {
            "content_id": content_id,
            "word_count": content_data.get("word_count", 0),
            "character_count": char_count,
            "character_count_no_spaces": char_count_no_spaces,
            "line_count": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "paragraph_count": paragraphs,
            "heading_count": len(headings),
            "link_count": len(links),
            "code_block_count": len(code_blocks),
            "headings": headings,
            "links": links,
            "code_blocks": code_blocks,
            "reading_time": reading_time,
            "content_hash": content_data.get("content_hash"),
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing markdown content {content_id}: {e}")
        return {}

# ============================================================================
# MARKDOWN CONTENT STATISTICS AND REPORTING
# ============================================================================

async def get_markdown_statistics(user_id: str, folder_id: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive markdown statistics for a user"""
    try:
        db = get_database_service()
        
        # Build query
        query = db.client.from_("markdown_content").select("word_count, created_at, files(folder_id)").eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("files.folder_id", folder_id)
            
        response = await query.execute()
        
        contents = response.data or []
        
        if not contents:
            return {
                "total_documents": 0,
                "total_words": 0,
                "average_words": 0,
                "median_words": 0,
                "min_words": 0,
                "max_words": 0,
                "total_estimated_reading_time": 0
            }
        
        # Calculate statistics
        word_counts = [content.get("word_count", 0) for content in contents]
        total_words = sum(word_counts)
        
        # Sort for median calculation
        sorted_words = sorted(word_counts)
        median_words = sorted_words[len(sorted_words) // 2] if sorted_words else 0
        
        # Estimated reading time (200 words per minute)
        total_reading_minutes = total_words / 200
        
        return {
            "total_documents": len(contents),
            "total_words": total_words,
            "average_words": round(total_words / len(contents), 1) if contents else 0,
            "median_words": median_words,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "total_estimated_reading_time": {
                "minutes": int(total_reading_minutes),
                "hours": round(total_reading_minutes / 60, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting markdown statistics for user {user_id}: {e}")
        return {}

async def get_word_count_distribution(user_id: str, folder_id: Optional[str] = None) -> Dict[str, int]:
    """Get word count distribution for user's markdown content"""
    try:
        db = get_database_service()
        
        # Build query
        query = db.client.from_("markdown_content").select("word_count").eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("files.folder_id", folder_id)
            
        response = await query.execute()
        
        contents = response.data or []
        
        # Create distribution buckets
        distribution = {
            "0-100": 0,
            "101-500": 0,
            "501-1000": 0,
            "1001-2000": 0,
            "2001-5000": 0,
            "5000+": 0
        }
        
        for content in contents:
            word_count = content.get("word_count", 0)
            
            if word_count <= 100:
                distribution["0-100"] += 1
            elif word_count <= 500:
                distribution["101-500"] += 1
            elif word_count <= 1000:
                distribution["501-1000"] += 1
            elif word_count <= 2000:
                distribution["1001-2000"] += 1
            elif word_count <= 5000:
                distribution["2001-5000"] += 1
            else:
                distribution["5000+"] += 1
        
        return distribution
        
    except Exception as e:
        logger.error(f"Error getting word count distribution for user {user_id}: {e}")
        return {}

async def get_content_creation_timeline(user_id: str, days: int = 30) -> List[Dict[str, Any]]:
    """Get timeline of content creation for a user"""
    try:
        db = get_database_service()
        
        # Calculate time threshold
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        response = await db.client.from_("markdown_content").select(
            "created_at, word_count, files(original_filename)"
        ).eq("user_id", user_id).gte("created_at", threshold_date).order("created_at", desc=True).execute()
        
        contents = response.data or []
        
        # Group by date
        timeline = {}
        for content in contents:
            created_at = content.get("created_at", "")
            date_str = created_at[:10]  # Extract date part
            
            if date_str not in timeline:
                timeline[date_str] = {
                    "date": date_str,
                    "documents": 0,
                    "total_words": 0
                }
            
            timeline[date_str]["documents"] += 1
            timeline[date_str]["total_words"] += content.get("word_count", 0)
        
        # Convert to list and sort
        timeline_list = list(timeline.values())
        timeline_list.sort(key=lambda x: x["date"])
        
        return timeline_list
        
    except Exception as e:
        logger.error(f"Error getting content creation timeline for user {user_id}: {e}")
        return []

# ============================================================================
# MARKDOWN CONTENT DEDUPLICATION AND CLEANUP
# ============================================================================

async def find_duplicate_content(user_id: str, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find duplicate markdown content based on hash"""
    try:
        db = get_database_service()
        
        # Build query
        query = db.client.from_("markdown_content").select(
            "id, file_id, content_hash, word_count, created_at, files(original_filename, folder_id)"
        ).eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("files.folder_id", folder_id)
            
        response = await query.execute()
        
        contents = response.data or []
        
        # Group by content hash
        hash_groups = {}
        for content in contents:
            content_hash = content.get("content_hash")
            if content_hash:
                if content_hash not in hash_groups:
                    hash_groups[content_hash] = []
                hash_groups[content_hash].append(content)
        
        # Find duplicates
        duplicates = []
        for content_hash, group in hash_groups.items():
            if len(group) > 1:
                duplicates.append({
                    "content_hash": content_hash,
                    "duplicate_count": len(group),
                    "contents": group
                })
        
        return duplicates
        
    except Exception as e:
        logger.error(f"Error finding duplicate content for user {user_id}: {e}")
        return []

async def remove_duplicate_content(user_id: str, folder_id: Optional[str] = None, keep_latest: bool = True) -> int:
    """Remove duplicate markdown content, keeping the latest or oldest"""
    try:
        duplicates = await find_duplicate_content(user_id, folder_id)
        
        removed_count = 0
        
        for duplicate_group in duplicates:
            contents = duplicate_group["contents"]
            
            # Sort by created_at
            contents.sort(key=lambda x: x.get("created_at", ""), reverse=keep_latest)
            
            # Keep the first one (latest or oldest based on sort), remove others
            to_remove = contents[1:]
            
            for content in to_remove:
                content_id = content["id"]
                file_id = content["file_id"]
                
                if await delete_markdown_content(file_id, user_id):
                    removed_count += 1
                    logger.info(f"Removed duplicate content {content_id}")
        
        logger.info(f"Removed {removed_count} duplicate content records for user {user_id}")
        return removed_count
        
    except Exception as e:
        logger.error(f"Error removing duplicate content for user {user_id}: {e}")
        return 0

async def cleanup_orphaned_content(user_id: str) -> int:
    """Clean up markdown content that references non-existent files"""
    try:
        db = get_database_service()
        
        # Get all markdown content for the user
        content_response = await db.client.from_("markdown_content").select("id, file_id").eq("user_id", user_id).execute()
        
        if not content_response.data:
            return 0
        
        # Get all valid file IDs
        files_response = await db.client.from_("files").select("id").eq("user_id", user_id).execute()
        valid_file_ids = set(file["id"] for file in files_response.data) if files_response.data else set()
        
        # Find orphaned content
        orphaned_content = []
        for content in content_response.data:
            if content["file_id"] not in valid_file_ids:
                orphaned_content.append(content["id"])
        
        # Delete orphaned content
        if orphaned_content:
            await db.client.from_("markdown_content").delete().in_("id", orphaned_content).execute()
            logger.info(f"Cleaned up {len(orphaned_content)} orphaned content records for user {user_id}")
            return len(orphaned_content)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error cleaning up orphaned content for user {user_id}: {e}")
        return 0

# ============================================================================
# MARKDOWN CONTENT EXPORT AND FORMATTING
# ============================================================================

async def export_markdown_content(
    user_id: str,
    folder_id: Optional[str] = None,
    export_format: str = "markdown",
    include_metadata: bool = True
) -> str:
    """Export markdown content in various formats"""
    try:
        # Get content
        contents = await get_markdown_contents_by_user(user_id, folder_id, limit=1000)
        
        if export_format == "markdown":
            return _export_as_markdown(contents, include_metadata)
        elif export_format == "html":
            return _export_as_html(contents, include_metadata)
        elif export_format == "text":
            return _export_as_text(contents, include_metadata)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
    except Exception as e:
        logger.error(f"Error exporting markdown content: {e}")
        return ""

def _export_as_markdown(contents: List[Dict[str, Any]], include_metadata: bool) -> str:
    """Export content as markdown format"""
    output = []
    
    for content in contents:
        if include_metadata:
            file_info = content.get("files", {})
            output.append(f"# {file_info.get('original_filename', 'Unknown File')}")
            output.append(f"**Word Count:** {content.get('word_count', 0)}")
            output.append(f"**Created:** {content.get('created_at', 'Unknown')}")
            output.append("")
        
        output.append(content.get("content", ""))
        output.append("")
        output.append("---")
        output.append("")
    
    return "\n".join(output)

def _export_as_html(contents: List[Dict[str, Any]], include_metadata: bool) -> str:
    """Export content as HTML format"""
    output = ["<!DOCTYPE html>", "<html>", "<head>", "<title>Markdown Content Export</title>", "</head>", "<body>"]
    
    for content in contents:
        if include_metadata:
            file_info = content.get("files", {})
            output.append(f"<h1>{file_info.get('original_filename', 'Unknown File')}</h1>")
            output.append(f"<p><strong>Word Count:</strong> {content.get('word_count', 0)}</p>")
            output.append(f"<p><strong>Created:</strong> {content.get('created_at', 'Unknown')}</p>")
        
        # Simple markdown to HTML conversion
        content_html = content.get("content", "").replace("\n", "<br>")
        output.append(f"<div>{content_html}</div>")
        output.append("<hr>")
    
    output.extend(["</body>", "</html>"])
    return "\n".join(output)

def _export_as_text(contents: List[Dict[str, Any]], include_metadata: bool) -> str:
    """Export content as plain text format"""
    output = []
    
    for content in contents:
        if include_metadata:
            file_info = content.get("files", {})
            output.append(f"FILE: {file_info.get('original_filename', 'Unknown File')}")
            output.append(f"WORD COUNT: {content.get('word_count', 0)}")
            output.append(f"CREATED: {content.get('created_at', 'Unknown')}")
            output.append("")
        
        # Remove markdown formatting
        text_content = re.sub(r'[#*_`\[\]()]+', '', content.get("content", ""))
        output.append(text_content)
        output.append("")
        output.append("=" * 80)
        output.append("")
    
    return "\n".join(output)
