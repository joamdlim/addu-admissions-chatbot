# backend/chatbot/test_pdf_to_docx_extraction.py

import os
import io
import tempfile
import re
from typing import List, Dict
from PyPDF2 import PdfReader
from chatbot.supabase_client import get_supabase_client

# Try to import pdf2docx and python-docx
try:
    from pdf2docx import Converter
    PDF2DOCX_AVAILABLE = True
    print("✅ pdf2docx available - can convert PDF to DOCX")
except ImportError:
    PDF2DOCX_AVAILABLE = False
    print("⚠️ pdf2docx not available")

try:
    from docx import Document
    PYTHON_DOCX_AVAILABLE = True
    print("✅ python-docx available - can extract from DOCX")
except ImportError:
    PYTHON_DOCX_AVAILABLE = False
    print("⚠️ python-docx not available")

if not PDF2DOCX_AVAILABLE or not PYTHON_DOCX_AVAILABLE:
    print("❌ Please install required libraries:")
    print("   pip install pdf2docx python-docx")

PDF_BUCKET = "original-pdfs"

def extract_text_with_pypdf2(pdf_bytes: bytes) -> str:
    """
    Standard PyPDF2 extraction (for comparison)
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
        return text
    except Exception as e:
        return f"PyPDF2 error: {e}"

def extract_text_via_docx(pdf_bytes: bytes) -> str:
    """
    Extract text by converting PDF to DOCX first, then extracting from DOCX
    """
    if not PDF2DOCX_AVAILABLE or not PYTHON_DOCX_AVAILABLE:
        return "Required libraries not available"
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_temp:
        pdf_temp.write(pdf_bytes)
        pdf_temp_path = pdf_temp.name
    
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as docx_temp:
        docx_temp_path = docx_temp.name
    
    try:
        print("🔄 Converting PDF to DOCX...")
        
        # Convert PDF to DOCX
        cv = Converter(pdf_temp_path)
        cv.convert(docx_temp_path, start=0, end=None)
        cv.close()
        
        print("✅ PDF converted to DOCX")
        print("🔄 Extracting text from DOCX...")
        
        # Extract text from DOCX
        doc = Document(docx_temp_path)
        text = ""
        
        page_num = 1
        text += f"\n\n--- Page {page_num} ---\n"
        
        for paragraph in doc.paragraphs:
            para_text = paragraph.text.strip()
            if para_text:
                text += para_text + "\n"
        
        print("✅ Text extracted from DOCX")
        return text
        
    except Exception as e:
        return f"PDF to DOCX conversion error: {e}"
    
    finally:
        # Clean up temporary files
        try:
            os.unlink(pdf_temp_path)
            os.unlink(docx_temp_path)
        except:
            pass

def clean_pdf_text(text: str) -> str:
    """
    Light cleaning since DOCX should preserve better structure
    """
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n[ \t]+', '\n', text)  # Remove spaces after newlines
    text = re.sub(r'[ \t]+\n', '\n', text)  # Remove spaces before newlines
    
    # Clean up step formatting
    text = re.sub(r'STEP\s*(\d+)\s*:\s*', r'\n\nSTEP \1: ', text)
    
    # Clean up multiple newlines but preserve intentional breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    
    return text.strip()

def analyze_step_order(text: str) -> Dict:
    """
    Analyze the order of steps in the text
    """
    step_pattern = r'STEP\s+(\d+):'
    matches = re.findall(step_pattern, text, re.IGNORECASE)
    
    if not matches:
        return {"found": False, "steps": [], "order": "No steps found"}
    
    step_numbers = [int(match) for match in matches]
    expected_order = list(range(1, max(step_numbers) + 1))
    
    analysis = {
        "found": True,
        "steps": step_numbers,
        "expected": expected_order,
        "correct_order": step_numbers == expected_order,
        "order": "Correct" if step_numbers == expected_order else "Incorrect"
    }
    
    return analysis

def smart_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Chunk text intelligently by preserving logical sections
    """
    chunks = []
    
    # Split by STEP sections first
    step_sections = re.split(r'(\n\n?STEP\s+\d+:)', text, flags=re.IGNORECASE)
    
    current_chunk = ""
    chunk_id = 0
    
    for i, section in enumerate(step_sections):
        if not section.strip():
            continue
        
        # Check if this is a STEP header
        if re.match(r'\n\n?STEP\s+\d+:', section, re.IGNORECASE):
            # This is a step header, combine with next section if available
            if i + 1 < len(step_sections):
                full_section = section + step_sections[i + 1]
                step_sections[i + 1] = ""  # Mark as processed
            else:
                full_section = section
        else:
            full_section = section
        
        if not full_section.strip():
            continue
            
        # If adding this section would exceed chunk size, finalize current chunk
        if len(current_chunk + "\n\n" + full_section) > chunk_size and current_chunk:
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'content': current_chunk.strip(),
                'size': len(current_chunk),
                'type': 'section'
            })
            chunk_id += 1
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                # Try to start overlap at a good break point
                for break_char in ['\n\n', '\n', '. ']:
                    break_pos = overlap_text.find(break_char)
                    if break_pos > 50:
                        overlap_text = overlap_text[break_pos + len(break_char):]
                        break
                current_chunk = overlap_text + "\n\n" + full_section
            else:
                current_chunk = full_section
        else:
            if current_chunk:
                current_chunk += "\n\n" + full_section
            else:
                current_chunk = full_section
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            'id': f'chunk_{chunk_id}',
            'content': current_chunk.strip(),
            'size': len(current_chunk),
            'type': 'section'
        })
    
    return chunks

def extract_and_chunk_pdf(pdf_bytes: bytes, chunk_size: int = 1000, overlap: int = 200) -> tuple:
    """
    Extract text from PDF using multiple methods and chunk it
    Returns: (pypdf2_text, docx_text, cleaned_text, chunks, analysis)
    """
    # Extract with PyPDF2 (for comparison)
    pypdf2_text = extract_text_with_pypdf2(pdf_bytes)
    
    # Extract via PDF->DOCX conversion
    docx_text = extract_text_via_docx(pdf_bytes)
    
    # Choose the best text for processing
    if PDF2DOCX_AVAILABLE and PYTHON_DOCX_AVAILABLE and "error" not in docx_text.lower():
        text_to_process = docx_text
        print("🔧 Using PDF->DOCX text for processing")
    else:
        text_to_process = pypdf2_text
        print("🔧 Using PyPDF2 text for processing")
    
    # Clean the text
    cleaned_text = clean_pdf_text(text_to_process)
    
    # Analyze step order
    analysis = analyze_step_order(cleaned_text)
    
    # Chunk the text
    chunks = smart_chunk_text(cleaned_text, chunk_size, overlap)
    
    return pypdf2_text, docx_text, cleaned_text, chunks, analysis

def main():
    if not PDF2DOCX_AVAILABLE or not PYTHON_DOCX_AVAILABLE:
        print("⚠️ For PDF->DOCX conversion, install:")
        print("   pip install pdf2docx python-docx")
        print("   Continuing with PyPDF2 only...\n")
    
    supabase = get_supabase_client()
    
    try:
        pdf_files = supabase.storage.from_(PDF_BUCKET).list()
        pdf_names = [f['name'] for f in pdf_files if f['name'].lower().endswith('.pdf')]
        
        if not pdf_names:
            print("❌ No PDF files found in the bucket.")
            return
        
        print("📄 Available PDFs:")
        for i, pdf_name in enumerate(pdf_names, 1):
            print(f"  {i}. {pdf_name}")
        
        # Let user select a PDF
        while True:
            try:
                choice = input(f"\nSelect a PDF (1-{len(pdf_names)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(pdf_names):
                    selected_pdf = pdf_names[choice_num - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(pdf_names)}")
            except ValueError:
                print("Please enter a valid number or 'q'")
        
        print(f"\n🔄 Processing: {selected_pdf}")
        
        # Download the PDF
        pdf_bytes = supabase.storage.from_(PDF_BUCKET).download(selected_pdf)
        
        # Extract and process
        pypdf2_text, docx_text, cleaned_text, chunks, analysis = extract_and_chunk_pdf(pdf_bytes)
        
        print("\n" + "="*80)
        print("📋 EXTRACTION COMPARISON")
        print("="*80)
        
        print(f"\n📊 STATISTICS:")
        print(f"   PyPDF2 text length: {len(pypdf2_text)} characters")
        if PDF2DOCX_AVAILABLE and PYTHON_DOCX_AVAILABLE:
            print(f"   PDF->DOCX text length: {len(docx_text)} characters")
        print(f"   Cleaned text length: {len(cleaned_text)} characters")
        print(f"   Number of chunks: {len(chunks)}")
        
        # Step order analysis
        print(f"\n🔍 STEP ORDER ANALYSIS:")
        if analysis["found"]:
            print(f"   Steps found: {analysis['steps']}")
            print(f"   Expected order: {analysis['expected']}")
            print(f"   Order status: {analysis['order']}")
        else:
            print(f"   {analysis['order']}")
        
        # Show PyPDF2 results
        pypdf2_analysis = analyze_step_order(pypdf2_text)
        print(f"\n📝 PyPDF2 EXTRACTED TEXT (first 800 chars):")
        print(f"   Step order: {pypdf2_analysis['order']}")
        print("-" * 50)
        print(pypdf2_text[:800])
        print("..." if len(pypdf2_text) > 800 else "")
        
        # Show PDF->DOCX results
        if PDF2DOCX_AVAILABLE and PYTHON_DOCX_AVAILABLE and "error" not in docx_text.lower():
            docx_analysis = analyze_step_order(docx_text)
            print(f"\n📝 PDF->DOCX EXTRACTED TEXT (first 800 chars):")
            print(f"   Step order: {docx_analysis['order']}")
            print("-" * 50)
            print(docx_text[:800])
            print("..." if len(docx_text) > 800 else "")
        
        print(f"\n🧹 FINAL PROCESSED TEXT (first 800 chars):")
        print("-" * 50)
        print(cleaned_text[:800])
        print("..." if len(cleaned_text) > 800 else "")
        
        # Ask if user wants to see full text
        show_full = input(f"\n🔍 Show full processed text? (y/N): ").strip().lower()
        if show_full == 'y':
            print(f"\n📄 FULL PROCESSED TEXT:")
            print("="*80)
            print(cleaned_text)
            print("="*80)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
