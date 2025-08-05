# backend/chatbot/test_supabase_pdf_scrape.py

import os
import io
from PyPDF2 import PdfReader
from chatbot.supabase_client import get_supabase_client

PDF_BUCKET = "original-pdfs"
TEXT_BUCKET = "scraped-texts"

def extract_text_from_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def main():
    supabase = get_supabase_client()

    # List all PDFs in the original-pdfs bucket
    pdf_files = supabase.storage.from_(PDF_BUCKET).list()
    print("PDFs found in bucket:", [f['name'] for f in pdf_files if f['name'].endswith('.pdf')])

    # For testing, pick the first PDF
    pdf_file = next((f['name'] for f in pdf_files if f['name'].endswith('.pdf')), None)
    if not pdf_file:
        print("No PDF files found in the bucket.")
        return

    print(f"Processing: {pdf_file}")

    # Download the PDF
    pdf_bytes = supabase.storage.from_(PDF_BUCKET).download(pdf_file)

    # Extract text
    text = extract_text_from_pdf(pdf_bytes)
    print("Extracted text preview:\n", text[:500], "\n---\n")

    # Upload the text to scraped-texts bucket
    text_filename = os.path.splitext(pdf_file)[0] + ".txt"
    supabase.storage.from_(TEXT_BUCKET).upload(
        text_filename,
        text.encode("utf-8"),
        {"content-type": "text/plain"}
    )
    print(f"✅ Uploaded extracted text as {text_filename} to {TEXT_BUCKET} bucket.")

if __name__ == "__main__":
    main()