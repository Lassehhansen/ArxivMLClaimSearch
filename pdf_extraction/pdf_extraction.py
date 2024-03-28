import os
import sys
import fitz 
import pypdfium2 as pdfium
import PyPDF2

def extract_text_pymupdf(filepath, page_limit=None):
    try:
        doc = fitz.open(filepath)
        text = ''
        for num, page in enumerate(doc):
            if page_limit is not None and num >= page_limit:
                break
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
    return None

def extract_text_pypdfium2(filepath, page_limit=None):
    try:
        doc = pdfium.PdfDocument(filepath)
        text = ''
        for num in range(len(doc)):
            if page_limit is not None and num >= page_limit:
                break
            page = doc.get_page(num)
            text_page = page.get_textpage()
            text += text_page.get_text_bounded()
        return text
    except Exception as e:
        print(f"Error with pypdfium2: {e}")
    return None

def extract_text_pypdf2(filepath, page_limit=None):
    try:
        pdf_file = open(filepath, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            if page_limit is not None and page_num >= page_limit:
                break
            page = pdf_reader.pages[page_num]
            text += page.extract_text()        
        return text
    except Exception as e:
        print(f"Error with PyPDF2: {e}")
    return None

def extract_text(filepath, page_limit = 10):
    for extractor in [extract_text_pymupdf, extract_text_pypdfium2, extract_text_pypdf2]:
        text = extractor(filepath, page_limit = page_limit)
        if text is not None:
            return text
    return ''