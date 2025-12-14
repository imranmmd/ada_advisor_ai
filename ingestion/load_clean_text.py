try:
    import pymupdf as fitz
except ImportError:
    import fitz

if not hasattr(fitz, "open"):
    raise ImportError(
        "PyMuPDF is required. Install it with `pip install PyMuPDF` "
        "(ensure the old `fitz` package is not installed)."
    )
import os
import re
from dotenv import load_dotenv

# Load environment variables from project .env for CLI execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# --------------------------------------------
# |          TEXT CLEANING TOOLS             |
# --------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize whitespace, remove duplicate blank lines,
    fix unicode, remove repeated spaces.
    """
    text = text.replace("\xa0", " ")              # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text) # collapse multi-blank lines
    return text.strip()


# Detect ANY kind of header/footer/page number
PAGE_BOUNDARY_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        =+\s*page\s*\d+\s*=+ |     # ===== PAGE 1 =====
        -+\s*page\s*\d+\s*-+ |     # ----- PAGE 1 -----
        \d+\s*\|\s*p\s*a\s*g\s*e | # 1 | P a g e
        page\s*\d+\s*(?:of\s*\d+)?|# Page 1 or Page 1 of 4
        \(?\s*\d+\s*\)? |          # (1) or 1
        \d+\s*/\s*\d+              # 1/4
    )
    \s*$
    """,
    flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def normalize_pages(text: str) -> str:
    """
    Strip inconsistent page markers/footers and re-add stable markers: === PAGE X ===.
    """

    # Remove standard metadata headers (ADA formats)
    text = re.sub(r"Doc No:\s*.*", "", text)
    text = re.sub(r"Version:\s*.*", "", text)
    text = re.sub(r"Date Effective:\s*.*", "", text)

    # Remove all page-like markers
    raw_pages = PAGE_BOUNDARY_PATTERN.split(text)
    pages = [p.strip() for p in raw_pages if p.strip()]

    # Fallback: treat whole document as one page
    if not pages:
        pages = [text.strip()]

    formatted_pages = []
    for idx, page_text in enumerate(pages, start=1):
        formatted_pages.append(
            f"=== PAGE {idx} ===\n{page_text.strip()}\n"
        )

    final_text = "\n".join(formatted_pages)

    # Clean excessive spacing after formatting
    final_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", final_text)

    return final_text.strip()


# --------------------------------------------
# | PDF → TXT CONVERSION + NORMALIZATION     |
# --------------------------------------------

def pdf_to_clean_txt(pdf_path, txt_path):
    """
    Converts a PDF → raw text → cleaned text → normalized pages → final TXT.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ ERROR opening PDF {pdf_path}: {e}")
        return

    all_pages_raw = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        if not text or text.strip() == "":
            print(f"⚠️ Warning: Page {page_number} is empty in {pdf_path}")
            continue

        cleaned = clean_text(text)
        all_pages_raw.append(cleaned)

    doc.close()

    raw_text = "\n\n".join(all_pages_raw)
    normalized_text = normalize_pages(raw_text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(normalized_text)

    print(f"✅ CLEAN TXT Saved: {txt_path}")


# --------------------------------------------
# | PROCESS ALL PDF FILES                    |
# --------------------------------------------

def convert_all_pdfs(
    documents_folder=None,
    output_folder="data/cleaned_text/"
):
    """
    Convert all PDFs in the documents_folder to cleaned TXT files in output_folder.
    """
    if documents_folder is None:
        documents_folder = os.getenv("ADA_PDF_SOURCE_DIR")

    if not documents_folder:
        raise RuntimeError(
            "Set ADA_PDF_SOURCE_DIR to the directory containing PDFs to process."
        )

    documents_folder = os.path.abspath(documents_folder)
    if not os.path.isdir(documents_folder):
        raise FileNotFoundError(
            f"Documents folder does not exist or is not a directory: {documents_folder}"
        )

    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(documents_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found in the documents folder.")
        return

    for filename in pdf_files:
        pdf_path = os.path.join(documents_folder, filename)
        txt_filename = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(output_folder, txt_filename)

        pdf_to_clean_txt(pdf_path, txt_path)
        print(f"➡️ Converted: {pdf_path} → {txt_path}")


# --------------------------------------------
# |                     RUN                  |
# --------------------------------------------

if __name__ == "__main__":
    convert_all_pdfs()
