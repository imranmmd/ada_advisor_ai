import fitz  # PyMuPDF
import os
import re

def clean_text(text: str) -> str:
    """
    Normalize whitespace, remove duplicate blank lines,
    fix unicode, remove repeated spaces.
    """
    text = text.replace("\xa0", " ")              # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text) # collapse multi-blank lines
    return text.strip()


def pdf_to_txt(pdf_path, txt_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ ERROR opening PDF {pdf_path}: {e}")
        return

    with open(txt_path, "w", encoding="utf-8") as f:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")

            if not text or text.strip() == "":
                print(f"⚠️ Warning: Page {page_number} is empty in {pdf_path}")
                continue

            cleaned = clean_text(text)

            f.write(f"\n\n===== PAGE {page_number} =====\n")
            f.write(cleaned)
            f.write("\n")

    doc.close()
    print(f"✅ Saved TXT: {txt_path}")


def convert_all_pdfs(
    documents_folder="/Users/macbook/Desktop/RAG Documents/All (Newest of each document)",
    output_folder="data/cleaned_text/"
):
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(documents_folder) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("⚠️ No PDF files found in the documents folder.")
        return

    for filename in pdf_files:
        pdf_path = os.path.join(documents_folder, filename)
        txt_filename = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(output_folder, txt_filename)

        pdf_to_txt(pdf_path, txt_path)
        print(f"Converted: {pdf_path} → {txt_path}")

# Run conversion
convert_all_pdfs()
