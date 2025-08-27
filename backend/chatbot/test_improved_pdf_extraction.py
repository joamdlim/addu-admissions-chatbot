# backend/chatbot/test_improved_pdf_extraction.py

import os
import io
import re
from typing import List, Dict
from chatbot.supabase_client import get_supabase_client

# Try different PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    print("✅ pdfplumber available - best for reading order")
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
    print("✅ PyPDF2 available - fallback option")
except ImportError:
    PYPDF2_AVAILABLE = False

if not PDFPLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
    print("❌ Please install a PDF library:")
    print("   pip install pdfplumber  # Recommended")
    print("   pip install PyPDF2      # Fallback")
    exit(1)

PDF_BUCKET = "original-pdfs"

def extract_text_with_pdfplumber(pdf_bytes: bytes) -> str:
    """
    Extract text using pdfplumber (better reading order)
    """
    if not PDFPLUMBER_AVAILABLE:
        return "pdfplumber not available"
    
    text = ""
    # pdfplumber needs a file-like object
    pdf_file = io.BytesIO(pdf_bytes)
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
    except Exception as e:
        return f"pdfplumber error: {e}"
    
    return text

def extract_text_with_pypdf2(pdf_bytes: bytes) -> str:
    """
    Extract text using PyPDF2 (fallback)
    """
    if not PYPDF2_AVAILABLE:
        return "PyPDF2 not available"
    
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

def clean_pdf_text(text: str) -> str:
    """
    Light cleaning - preserve structure since pdfplumber should give good order
    """
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n[ \t]+', '\n', text)  # Remove spaces after newlines
    text = re.sub(r'[ \t]+\n', '\n', text)  # Remove spaces before newlines
    
    # Fix minor formatting issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    
    # Clean up step formatting - make it more consistent
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
    Extract text from PDF and chunk it intelligently
    Returns: (pypdf2_text, pdfplumber_text, cleaned_text, chunks, analysis)
    """
    # Extract with both methods for comparison
    pypdf2_text = extract_text_with_pypdf2(pdf_bytes)
    pdfplumber_text = extract_text_with_pdfplumber(pdf_bytes)
    
    # Choose the best text for processing
    if PDFPLUMBER_AVAILABLE and "error" not in pdfplumber_text.lower():
        text_to_process = pdfplumber_text
        print("🔧 Using pdfplumber text for processing")
    elif PYPDF2_AVAILABLE and "error" not in pypdf2_text.lower():
        text_to_process = pypdf2_text
        print("🔧 Using PyPDF2 text for processing")
    else:
        raise RuntimeError("No working PDF extraction method available")
    
    # Clean the text
    cleaned_text = clean_pdf_text(text_to_process)
    
    # Analyze step order
    analysis = analyze_step_order(cleaned_text)
    
    # Chunk the text
    chunks = smart_chunk_text(cleaned_text, chunk_size, overlap)
    
    return pypdf2_text, pdfplumber_text, cleaned_text, chunks, analysis

def main():
    if not PDFPLUMBER_AVAILABLE:
        print("⚠️ For best results, install pdfplumber:")
        print("   pip install pdfplumber")
        print("   Continuing with PyPDF2...\n")
    
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
        pypdf2_text, pdfplumber_text, cleaned_text, chunks, analysis = extract_and_chunk_pdf(pdf_bytes)
        
        print("\n" + "="*80)
        print("📋 EXTRACTION COMPARISON")
        print("="*80)
        
        print(f"\n📊 STATISTICS:")
        if PYPDF2_AVAILABLE:
            print(f"   PyPDF2 text length: {len(pypdf2_text)} characters")
        if PDFPLUMBER_AVAILABLE:
            print(f"   pdfplumber text length: {len(pdfplumber_text)} characters")
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
        
        if PYPDF2_AVAILABLE and "error" not in pypdf2_text.lower():
            pypdf2_analysis = analyze_step_order(pypdf2_text)
            print(f"\n📝 PyPDF2 EXTRACTED TEXT (first 800 chars):")
            print(f"   Step order: {pypdf2_analysis['order']}")
            print("-" * 50)
            print(pypdf2_text[:800])
            print("..." if len(pypdf2_text) > 800 else "")
        
        if PDFPLUMBER_AVAILABLE and "error" not in pdfplumber_text.lower():
            plumber_analysis = analyze_step_order(pdfplumber_text)
            print(f"\n📝 pdfplumber EXTRACTED TEXT (first 800 chars):")
            print(f"   Step order: {plumber_analysis['order']}")
            print("-" * 50)
            print(pdfplumber_text[:800])
            print("..." if len(pdfplumber_text) > 800 else "")
        
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
