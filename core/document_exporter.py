import io
import re
from datetime import datetime
from typing import List, Dict, Any
from loguru import logger
from docx import Document
from docx.shared import RGBColor, Pt
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from bleach import clean


logger.add("logs/document_exporter.log", rotation="100 MB", retention="7 days", level="INFO")


WHITELIST_TAGS = {
    'div', 'span', 'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'thead', 'tbody'
}


class DocumentExporter:
    """Handle document export and preview generation with enhanced formatting"""
    
    def __init__(self):
        self.logger = logger
    
    # def generate_html_preview(self, template_content: str, clauses: List[Dict[str, Any]]) -> str:
    #     """Generate HTML preview with color-coded elements and XSS protection"""
    #     self.logger.info("Generating HTML preview")
    #     formatted_content = self._format_template_html(template_content)
        
    #     html = f"""
    #     <!DOCTYPE html>
    #     <html>
    #     <head>
    #         <title>Legal Template Preview</title>
    #         <style>
    #             body {{ 
    #                 font-family: 'Times New Roman', serif; 
    #                 line-height: 1.6; 
    #                 max-width: 900px; 
    #                 margin: 0 auto; 
    #                 padding: 20px;
    #                 background-color: #ffffff;
    #             }}
    #             .header {{
    #                 text-align: center;
    #                 border-bottom: 2px solid #333;
    #                 padding-bottom: 20px;
    #                 margin-bottom: 30px;
    #             }}
    #             /* RED placeholders */
    #             .placeholder {{ 
    #                 background-color: #fee2e2;
    #                 color: #dc2626;
    #                 padding: 3px 6px;
    #                 font-weight: bold;
    #                 border-radius: 3px;
    #                 border: 1px solid #dc2626;
    #             }}
    #             /* BLUE drafting notes */
    #             .drafting-note {{ 
    #                 color: #1976d2;
    #                 background-color: #e3f2fd;
    #                 padding: 10px;
    #                 margin: 12px 0;
    #                 border-left: 4px solid #1976d2;
    #                 border-radius: 4px;
    #                 font-size: 0.9em;
    #                 font-style: italic;
    #             }}
    #             /* GREEN alternative clauses */
    #             .alternative-clause {{
    #                 color: #059669;
    #                 background-color: #ecfdf5;
    #                 padding: 10px;
    #                 margin: 12px 0;
    #                 border-left: 4px solid #059669;
    #                 border-radius: 4px;
    #                 font-size: 0.9em;
    #             }}
    #             h1, h2, h3 {{ color: #1f2937; }}
    #         </style>
    #     </head>
    #     <body>
    #         <div class="header">
    #             <h1>Legal Contract Template</h1>
    #             <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
    #         </div>
            
    #         <div class="template-content">
    #             {formatted_content}
    #         </div>
    #     </body>
    #     </html>
    #     """
        
    #     # Clean HTML to prevent XSS
    #     safe_html = clean(html, tags=WHITELIST_TAGS, strip=True)
    #     return safe_html
    
    # def _format_template_html(self, content: str) -> str:
    #     """Format template content for HTML display with enhanced color coding"""
    #     import re
        
    #     # First, process the content to ensure drafting notes appear after full clauses
    #     # Split content into sections
    #     sections = re.split(r'(### \d+\.\d+\.)', content)
    #     formatted_sections = []
        
    #     for i in range(0, len(sections), 2):
    #         if i + 1 < len(sections):
    #             section_title = sections[i]
    #             section_content = sections[i + 1]
                
    #             # Extract drafting note if it exists
    #             drafting_note_match = re.search(r'\[DRAFTING NOTE: ([^\]]+)\]', section_content)
    #             if drafting_note_match:
    #                 # Remove drafting note from content
    #                 section_content = re.sub(r'\[DRAFTING NOTE: [^\]]+\]', '', section_content).strip()
    #                 drafting_note = drafting_note_match.group(1)
                    
    #                 # Combine title, content, and drafting note
    #                 formatted_section = f"{section_title}{section_content}\n\n<div class='drafting-note'><strong>📝 Drafting Note:</strong> {drafting_note}</div>"
    #             else:
    #                 formatted_section = f"{section_title}{section_content}"
                
    #             formatted_sections.append(formatted_section)
    #         else:
    #             formatted_sections.append(sections[i])
        
    #     # Join sections back together
    #     content = '\n\n'.join(formatted_sections)
        
    #     # Replace placeholders with RED highlighting
    #     content = re.sub(r'\[([^\]]+?)\](?!\s*:)', r'<span class="placeholder">[\1]</span>', content)
        
    #     # Replace alternative clauses with GREEN formatting
    #     content = re.sub(
    #         r'\[ALTERNATIVE CLAUSE: ([^\]]+)\]', 
    #         r'<div class="alternative-clause"><strong>🔄 Alternative Clause:</strong> \1</div>', 
    #         content,
    #         flags=re.DOTALL
    #     )
        
    #     # Convert line breaks to HTML
    #     content = content.replace('\n', '<br>')
        
    #     return content
    
    # def _generate_clause_database_html(self, clauses: List[Dict[str, Any]]) -> str:
    #     """Generate enhanced clause database section"""
    #     if not clauses:
    #         return ""
        
    #     # Group clauses by type
    #     from collections import defaultdict
    #     clause_groups = defaultdict(list)
        
    #     for clause in clauses:
    #         clause_groups[clause["clause_type"]].append(clause)
        
    #     html = '<div class="clause-database"><h2>📚 Comprehensive Clause Database</h2>'
    #     html += '<p><em>All alternative clauses extracted from uploaded contracts</em></p>'
        
    #     for clause_type, type_clauses in clause_groups.items():
    #         html += f'<h3>{clause_type.replace("_", " ").title()} Clauses ({len(type_clauses)} variations)</h3>'
            
    #         for i, clause in enumerate(type_clauses, 1):
    #             metadata = clause.get("contract_metadata", {})
    #             html += f'''
    #             <div class="clause-item">
    #                 <div class="clause-header">{clause_type.replace("_", " ").title()} Clause {i}</div>
    #                 <div class="metadata">
    #                     <strong>📄 Source Contract:</strong> {clause.get("source_contract", "Unknown")}<br>
    #                     <strong>👥 Parties:</strong> {", ".join(clause.get("contract_parties", ["Unknown"]))}<br>
    #                     <strong>📅 Date:</strong> {clause.get("contract_date", "Unknown")}<br>
    #                     <strong>📍 Position:</strong> {clause.get("position_context", "Unknown")}<br>
    #                     <strong>🎯 Purpose:</strong> {clause.get("clause_purpose", "Not specified")}
    #                 </div>
    #                 <div class="clause-text"><strong>Clause Text:</strong><br>{clause["clause_text"]}</div>
    #             </div>
    #             '''
        
    #     html += '</div>'
    #     return html
    
    def generate_docx(self, template_content: str) -> bytes:
        """Generate DOCX file with proper color formatting"""
        doc = Document()
        
        # Create custom styles
        styles = doc.styles
        
        # Create or get list styles
        try:
            list_style_1 = styles['List Number']
        except KeyError:
            list_style_1 = styles.add_style('List Number', WD_STYLE_TYPE.PARAGRAPH)
            # list_style_1.base_style = styles['Normal']
        
        try:
            list_style_2 = styles['List Number 2']
        except KeyError:
            list_style_2 = styles.add_style('List Number 2', WD_STYLE_TYPE.PARAGRAPH)
            # list_style_2.base_style = styles['Normal']
        
        # Add title
        title = doc.add_heading('Legal Contract Template', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add generation info
        info_para = doc.add_paragraph(f'Generated on {datetime.now().strftime("%B %d, %Y")}')
        info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_paragraph()  # Add space
        
        # Add legend for color coding and formatting
        legend_para = doc.add_paragraph()
        legend_para.add_run("Formatting Guide: ").bold = True
        
        # RED for placeholders
        red_run = legend_para.add_run("[Placeholders]")
        red_run.font.color.rgb = RGBColor(220, 38, 38)  # Red
        red_run.font.highlight_color = WD_COLOR_INDEX.PINK
        red_run.bold = True
        
        legend_para.add_run(" | ")
        
        # BLUE for drafting notes
        blue_run = legend_para.add_run("Drafting Notes")
        blue_run.font.color.rgb = RGBColor(25, 118, 210)  # Blue
        blue_run.italic = True
        
        legend_para.add_run(" | ")
        
        # GREEN for alternatives
        green_run = legend_para.add_run("Alternative Clauses")
        green_run.font.color.rgb = RGBColor(5, 150, 105)  # Green
        
        # Add list formatting info
        list_info_para = doc.add_paragraph()
        list_info_para.add_run("📝 List Formatting: ").bold = True
        list_info_para.add_run("This template uses Microsoft Word's built-in list styles. You can easily insert, delete, or reorder items without manual renumbering!")
        
        doc.add_paragraph()  # Add space
        
        # Process template content with enhanced formatting
        self._add_enhanced_content_to_doc(doc, template_content)
        
        # Save to bytes
        doc_bytes = io.BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)
        
        return doc_bytes.getvalue()
    
    def _add_enhanced_content_to_doc(self, doc, content: str):
        """Add enhanced formatted content to DOCX document with proper list styles"""
        self.logger.info("Adding enhanced content to DOCX document")
        
        # Split content into individual lines to process each one correctly
        lines = content.split('\n')
        
        # Track current numbering for different list types
        current_main_number = 0  # For NUMBERED_LIST_ITEM
        current_sub_numbers = {0: 0}  # For SUB_NUMBERED_ITEM
        current_sub_sub_numbers = {0: {0: 0}}  # For SUB_SUB_NUMBERED_ITEM
        
        # For alphabetical items (a, b, c)
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        current_alpha_index = -1  # For ALPHA_ITEM_MAIN or ALPHA_ITEM
        current_sub_alpha_indices = {}  # For SUB_ALPHA_ITEM
        current_sub_sub_alpha_indices = {}  # For SUB_SUB_ALPHA_ITEM
        
        # For Roman numerals (i, ii, iii)
        roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 
                        'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx']
        current_roman_index = -1  # For ROMAN_ITEM_MAIN or ROMAN_ITEM
        current_sub_roman_indices = {}  # For SUB_ROMAN_ITEM
        current_sub_sub_roman_indices = {}  # For SUB_SUB_ROMAN_ITEM
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a heading
            if line.startswith('## '):
                heading_text = line.lstrip('# ').strip()
                doc.add_heading(heading_text, level=2)
                continue
            elif line.startswith('### '):
                heading_text = line.lstrip('# ').strip()
                doc.add_heading(heading_text, level=3)
                continue
            elif line.startswith('**') and line.endswith('**'):
                p = doc.add_paragraph()
                p.add_run(line.strip('*')).bold = True
                continue

            # Check for list items using a more comprehensive regex
            list_match = re.match(r'^(NUMBERED_LIST_ITEM|SUB_NUMBERED_ITEM|SUB_SUB_NUMBERED_ITEM|'
                                r'ALPHA_ITEM_MAIN|SUB_ALPHA_ITEM|SUB_SUB_ALPHA_ITEM|'
                                r'ROMAN_ITEM_MAIN|SUB_ROMAN_ITEM|SUB_SUB_ROMAN_ITEM|'
                                r'BULLET_ITEM|SUB_BULLET_ITEM|SUB_SUB_BULLET_ITEM|'
                                r'ALPHA_ITEM|ROMAN_ITEM):\s*(.*)', line)
            
            if list_match:
                list_type, item_content = list_match.groups()
                
                # Create paragraph
                p = doc.add_paragraph()
                
                # Handle different list types
                if list_type == 'NUMBERED_LIST_ITEM':
                    current_main_number += 1
                    # Reset sub-numbers for this main number
                    current_sub_numbers[current_main_number] = 0
                    current_sub_sub_numbers[current_main_number] = {}
                    
                    # Add the number prefix manually
                    p.add_run(f"{current_main_number}. ").bold = True
                    p.paragraph_format.left_indent = Pt(18)
                    
                elif list_type == 'SUB_NUMBERED_ITEM':
                    # If no main number exists yet, create one
                    if current_main_number == 0:
                        current_main_number = 1
                        current_sub_numbers[current_main_number] = 0
                        current_sub_sub_numbers[current_main_number] = {}
                    
                    # Increment the sub-number for the current main number
                    current_sub_numbers[current_main_number] += 1
                    current_sub = current_sub_numbers[current_main_number]
                    
                    # Reset sub-sub numbers for this sub number
                    current_sub_sub_numbers[current_main_number][current_sub] = 0
                    
                    # Add the number prefix manually
                    p.add_run(f"{current_main_number}.{current_sub} ").bold = True
                    p.paragraph_format.left_indent = Pt(36)
                    
                elif list_type == 'SUB_SUB_NUMBERED_ITEM':
                    # If no main number exists yet, create one
                    if current_main_number == 0:
                        current_main_number = 1
                        current_sub_numbers[current_main_number] = 0
                        current_sub_sub_numbers[current_main_number] = {}
                    
                    # If no sub number exists yet, create one
                    if current_sub_numbers[current_main_number] == 0:
                        current_sub_numbers[current_main_number] = 1
                        current_sub_sub_numbers[current_main_number][1] = 0
                    
                    current_sub = current_sub_numbers[current_main_number]
                    
                    # Increment the sub-sub-number
                    if current_sub not in current_sub_sub_numbers[current_main_number]:
                        current_sub_sub_numbers[current_main_number][current_sub] = 0
                        
                    current_sub_sub_numbers[current_main_number][current_sub] += 1
                    current_sub_sub = current_sub_sub_numbers[current_main_number][current_sub]
                    
                    # Add the number prefix manually
                    p.add_run(f"{current_main_number}.{current_sub}.{current_sub_sub} ").bold = True
                    p.paragraph_format.left_indent = Pt(54)
                    
                # Handle alphabetical items
                elif list_type in ('ALPHA_ITEM_MAIN', 'ALPHA_ITEM'):
                    current_alpha_index += 1
                    alpha_char = alphabet[current_alpha_index % len(alphabet)]
                    p.add_run(f"({alpha_char}) ").bold = True
                    p.paragraph_format.left_indent = Pt(18)
                    
                elif list_type == 'SUB_ALPHA_ITEM':
                    if current_main_number == 0:
                        current_main_number = 1
                    
                    if current_main_number not in current_sub_alpha_indices:
                        current_sub_alpha_indices[current_main_number] = -1
                        
                    current_sub_alpha_indices[current_main_number] += 1
                    alpha_index = current_sub_alpha_indices[current_main_number]
                    alpha_char = alphabet[alpha_index % len(alphabet)]
                    
                    p.add_run(f"({alpha_char}) ").bold = True
                    p.paragraph_format.left_indent = Pt(36)
                    
                elif list_type == 'SUB_SUB_ALPHA_ITEM':
                    if current_main_number == 0:
                        current_main_number = 1
                    
                    if current_sub_numbers[current_main_number] == 0:
                        current_sub_numbers[current_main_number] = 1
                        
                    current_sub = current_sub_numbers[current_main_number]
                    
                    key = f"{current_main_number}_{current_sub}"
                    if key not in current_sub_sub_alpha_indices:
                        current_sub_sub_alpha_indices[key] = -1
                        
                    current_sub_sub_alpha_indices[key] += 1
                    alpha_index = current_sub_sub_alpha_indices[key]
                    alpha_char = alphabet[alpha_index % len(alphabet)]
                    
                    p.add_run(f"({alpha_char}) ").bold = True
                    p.paragraph_format.left_indent = Pt(54)
                    
                # Handle Roman numeral items
                elif list_type in ('ROMAN_ITEM_MAIN', 'ROMAN_ITEM'):
                    current_roman_index += 1
                    roman_char = roman_numerals[current_roman_index % len(roman_numerals)]
                    p.add_run(f"({roman_char}) ").bold = True
                    p.paragraph_format.left_indent = Pt(18)
                    
                elif list_type == 'SUB_ROMAN_ITEM':
                    if current_main_number == 0:
                        current_main_number = 1
                    
                    if current_main_number not in current_sub_roman_indices:
                        current_sub_roman_indices[current_main_number] = -1
                        
                    current_sub_roman_indices[current_main_number] += 1
                    roman_index = current_sub_roman_indices[current_main_number]
                    roman_char = roman_numerals[roman_index % len(roman_numerals)]
                    
                    p.add_run(f"({roman_char}) ").bold = True
                    p.paragraph_format.left_indent = Pt(36)
                    
                elif list_type == 'SUB_SUB_ROMAN_ITEM':
                    if current_main_number == 0:
                        current_main_number = 1
                    
                    if current_sub_numbers[current_main_number] == 0:
                        current_sub_numbers[current_main_number] = 1
                        
                    current_sub = current_sub_numbers[current_main_number]
                    
                    key = f"{current_main_number}_{current_sub}"
                    if key not in current_sub_sub_roman_indices:
                        current_sub_sub_roman_indices[key] = -1
                        
                    current_sub_sub_roman_indices[key] += 1
                    roman_index = current_sub_sub_roman_indices[key]
                    roman_char = roman_numerals[roman_index % len(roman_numerals)]
                    
                    p.add_run(f"({roman_char}) ").bold = True
                    p.paragraph_format.left_indent = Pt(54)
                    
                # Handle bullet items
                elif list_type == 'BULLET_ITEM':
                    p.add_run("• ").bold = True  # Using Unicode bullet directly
                    p.paragraph_format.left_indent = Pt(18)
                    
                elif list_type == 'SUB_BULLET_ITEM':
                    p.add_run("◦ ").bold = True  # Using Unicode circle directly
                    p.paragraph_format.left_indent = Pt(36)
                    
                elif list_type == 'SUB_SUB_BULLET_ITEM':
                    p.add_run("▪ ").bold = True  # Using Unicode square directly
                    p.paragraph_format.left_indent = Pt(54)
                
                # Process the content without adding the number again
                self._process_content_for_doc(p, item_content)
                continue

            # Handle regular paragraphs that are not list items or headings
            p = doc.add_paragraph()
            self._process_content_for_doc(p, line)
    
    def _process_content_for_doc(self, paragraph, content: str):
        """Process content for DOCX with enhanced formatting including drafting notes and alternatives"""
        self.logger.debug(f"Processing content for DOCX: {content[:100]}")
        
        # Handle Drafting Notes
        drafting_note_match = re.search(r'\[DRAFTING NOTE: ([^\]]+)\]', content, re.DOTALL)
        if drafting_note_match:
            note_text = drafting_note_match.group(1)
            run = paragraph.add_run(f"📝 Drafting Note: {note_text}")
            run.font.color.rgb = RGBColor(25, 118, 210)  # Blue
            run.italic = True
            return  # Stop processing this line further

        # Handle Alternative Clauses
        alt_clause_match = re.search(r'\[ALTERNATIVE CLAUSE: ([^\]]+)\]', content, re.DOTALL)
        if alt_clause_match:
            alt_text = alt_clause_match.group(1)
            run = paragraph.add_run(f"🔄 Alternative Clause: {alt_text}")
            run.font.color.rgb = RGBColor(5, 150, 105)  # Green
            return  # Stop processing this line further

        # Process regular text with placeholders
        # Split the line by the placeholder pattern, keeping the delimiters
        parts = re.split(r'(\[[^\]]+?\](?!\s*:)(?!DRAFTING NOTE:)(?!ALTERNATIVE CLAUSE:))', content)
        for part in parts:
            if not part:
                continue
            
            # Check if the part is a placeholder (but not a drafting note or alternative clause)
            if part.startswith('[') and part.endswith(']') and 'DRAFTING NOTE:' not in part and 'ALTERNATIVE CLAUSE:' not in part:
                run = paragraph.add_run(part)
                run.font.color.rgb = RGBColor(220, 38, 38)  # Red
                run.font.highlight_color = WD_COLOR_INDEX.PINK
                run.bold = True
            else:
                # It's regular text
                paragraph.add_run(part)