import re

# Matches any line that looks like a page marker/header/footer in a variety of formats.
PAGE_BOUNDARY_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        =+\s*page\s*\d+\s*=+ |     # ===== PAGE 1 ===== or === PAGE 1 ===
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
    # Remove document metadata headers that can appear at the top of each page.
    text = re.sub(r"Doc No:\s*.*", "", text)
    text = re.sub(r"Version:\s*.*", "", text)
    text = re.sub(r"Date Effective:\s*.*", "", text)

    # Split on any page marker variant; the regex eats the marker itself.
    raw_pages = PAGE_BOUNDARY_PATTERN.split(text)
    pages = [p.strip() for p in raw_pages if p.strip()]

    # If no marker was found, treat the whole text as a single page.
    if not pages:
        pages = [text.strip()]

    formatted_pages = []
    for idx, page_text in enumerate(pages, start=1):
        formatted_pages.append(f"=== PAGE {idx} ===\n{page_text.strip()}\n")

    final_text = "\n".join(formatted_pages)
    final_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", final_text)  # collapse excess blank lines

    return final_text.strip()

with open("data/cleaned_text/_02_00_approved_core_competence_and_learning_outcome_specification_policy.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

cleaned_text = normalize_pages(raw_text)

with open("cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)