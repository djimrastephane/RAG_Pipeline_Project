import pandas as pd
from pathlib import Path
from collections import Counter

DOC_ID = "Grampian-2022-2023"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data_processed" / DOC_ID

# noinspection SpellCheckingInspection
PRIMARY_EXTRACTOR_KEY = "pymupdf"
# noinspection SpellCheckingInspection
FALLBACK_EXTRACTOR_KEY = "pdfplumber"


def validate_text_quality():
    """Comprehensive text extraction quality checks."""

    # Load pages
    pages_df = pd.read_parquet(OUT_DIR / "pages.parquet")

    print("=" * 70)
    print("TEXT EXTRACTION QUALITY REPORT")
    print("=" * 70)

    # 1. COMPLETENESS CHECK
    print("\n1. COMPLETENESS")
    print("-" * 70)

    empty_pages = pages_df[pages_df["clean_text"].str.len() < 50]
    print(f"Total pages: {len(pages_df)}")
    print(f"Empty pages (<50 chars): {len(empty_pages)}")

    if len(empty_pages) > 0:
        print(f"  ⚠️ Warning: Pages {list(empty_pages['page'].values)} are mostly empty")
    else:
        print(f"  ✅ All pages have substantial text")

    # 2. TEXT LENGTH DISTRIBUTION
    print("\n2. TEXT LENGTH DISTRIBUTION")
    print("-" * 70)

    pages_df['text_length'] = pages_df['clean_text'].str.len()
    stats = pages_df['text_length'].describe()

    print(f"  Min: {stats['min']:.0f} chars")
    print(f"  Average: {stats['mean']:.0f} chars")
    print(f"  Max: {stats['max']:.0f} chars")
    print(f"  Pages with <200 chars: {sum(pages_df['text_length'] < 200)}")

    # 3. ENCODING QUALITY
    print("\n3. ENCODING QUALITY")
    print("-" * 70)

    # Check for replacement characters (bad encoding)
    replacement_char = "\uFFFD"
    pages_with_replacement = sum(pages_df['clean_text'].str.contains(replacement_char, na=False))

    # Check alphabetic ratio
    def alpha_ratio(text):
        if not text:
            return 0
        alpha = sum(c.isalpha() for c in text)
        return alpha / len(text)

    pages_df['alpha_ratio'] = pages_df['clean_text'].apply(alpha_ratio)
    low_alpha = sum(pages_df['alpha_ratio'] < 0.3)

    print(f"  Pages with replacement chars (�): {pages_with_replacement}")
    print(f"  Pages with low alphabetic ratio (<30%): {low_alpha}")

    if pages_with_replacement > 0 or low_alpha > 5:
        print(f"  ⚠️ Potential encoding issues detected")
    else:
        print(f"  ✅ Text encoding looks good")

    # 4. EXTRACTOR PERFORMANCE
    print("\n4. EXTRACTOR PERFORMANCE")
    print("-" * 70)

    extractor_counts = pages_df['extractor'].value_counts()
    print(f"  PyMuPDF (primary): {extractor_counts.get(PRIMARY_EXTRACTOR_KEY, 0)} pages")
    print(f"  PDFPlumber (fallback): {extractor_counts.get(FALLBACK_EXTRACTOR_KEY, 0)} pages")

    # Check fallback reasons
    fallback_notes = pages_df[pages_df['extractor'] == FALLBACK_EXTRACTOR_KEY]['extractor_notes'].value_counts()
    if len(fallback_notes) > 0:
        print(f"\n  Fallback reasons:")
        for reason, count in fallback_notes.items():
            print(f"    - {reason}: {count} pages")

    # 5. SECTION DETECTION
    print("\n5. SECTION DETECTION")
    print("-" * 70)

    sections_df = pd.read_parquet(OUT_DIR / "sections.parquet")
    print(f"  Total sections detected: {len(sections_df)}")
    print(f"  Average pages per section: {len(pages_df) / len(sections_df):.1f}")

    # Show top sections
    print(f"\n  Top 5 sections by page count:")
    top_sections = sections_df.nlargest(5, 'word_count')[['section_title', 'page_start', 'page_end', 'word_count']]
    for _, s in top_sections.iterrows():
        print(
            f"    - {s['section_title'][:40]:<40} (p{s['page_start']:>3}-{s['page_end']:<3}): {s['word_count']:>6} words")

    # 6. SAMPLE TEXT QUALITY
    print("\n6. SAMPLE TEXT QUALITY")
    print("-" * 70)

    # Get a typical text page (not table, middle of document)
    text_pages = pages_df[pages_df['is_table'] == False]
    mid_page = text_pages.iloc[len(text_pages) // 2]

    print(f"\n  Sample from page {mid_page['page']} (middle of document):")
    print(f"  Section: {mid_page.get('section_title', 'Unknown')}")
    print(f"  Length: {len(mid_page['clean_text'])} chars")
    print(f"\n  Preview (first 500 chars):")
    print("-" * 70)
    print(mid_page['clean_text'][:500])
    print("-" * 70)

    # 7. BOILERPLATE REMOVAL
    print("\n7. BOILERPLATE REMOVAL")
    print("-" * 70)

    # Check if common headers/footers were removed
    all_text = " ".join(pages_df['clean_text'].tolist())

    common_phrases = {
        "NHS GRAMPIAN": all_text.count("NHS GRAMPIAN"),
        "ANNUAL REPORT": all_text.count("ANNUAL REPORT"),
        "Page": all_text.count("Page "),  # Page numbers
    }

    print(f"  Phrase frequency in cleaned text:")
    for phrase, count in common_phrases.items():
        frequency = count / len(pages_df)
        print(f"    '{phrase}': {count} occurrences ({frequency:.1f} per page)")

        if phrase in ["NHS GRAMPIAN", "ANNUAL REPORT"] and frequency > 0.5:
            print(f"      ⚠️ May be repeated header/footer (expected <0.5 per page)")

    # 8. OVERALL QUALITY SCORE
    print("\n8. OVERALL QUALITY ASSESSMENT")
    print("=" * 70)

    issues = []

    if len(empty_pages) > 5:
        issues.append(f"❌ Too many empty pages ({len(empty_pages)})")

    if pages_with_replacement > 10:
        issues.append(f"❌ Encoding issues on {pages_with_replacement} pages")

    if low_alpha > 10:
        issues.append(f"⚠️ Low text quality on {low_alpha} pages")

    if len(sections_df) < 10:
        issues.append(f"⚠️ Few sections detected ({len(sections_df)})")

    if len(issues) == 0:
        print("✅ EXCELLENT: Text extraction quality is high")
        print("✅ Ready for batch processing")
        return True
    else:
        print("⚠️ ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        print("\n⚠️ Review issues before batch processing")
        return False


def compare_with_source_pdf():
    """Compare a sample page with source PDF."""
    import pymupdf as fitz

    pdf_path = Path(
        PROJECT_ROOT
        / "Data"
        / "Annual Accounts NHS Grampian"
        / "Preliminary_Test"
        / "Grampian-2022-2023.pdf"
    )

    if not pdf_path.exists():
        print("\n⚠️ PDF not found, skipping source comparison")
        return

    pages_df = pd.read_parquet(OUT_DIR / "pages.parquet")

    print("\n9. SOURCE PDF COMPARISON")
    print("=" * 70)

    # Pick a text-heavy page (not table)
    text_pages = pages_df[pages_df['is_table'] == False]
    test_page = text_pages.iloc[10]  # Page 10 or so

    # Extract raw from PDF
    doc = fitz.open(pdf_path)
    raw_text = doc.load_page(test_page['page'] - 1).get_text("text")
    doc.close()

    # Compare lengths
    raw_len = len(raw_text)
    clean_len = len(test_page['clean_text'])
    reduction = (1 - clean_len / raw_len) * 100

    print(f"\n  Test page: {test_page['page']}")
    print(f"  Raw text length: {raw_len} chars")
    print(f"  Clean text length: {clean_len} chars")
    print(f"  Reduction: {reduction:.1f}% (boilerplate removed)")

    # Check if key content preserved
    sample_words = ['report', 'performance', 'financial', 'NHS', 'patient']
    preserved = []
    for word in sample_words:
        in_raw = word.lower() in raw_text.lower()
        in_clean = word.lower() in test_page['clean_text'].lower()
        if in_raw:
            preserved.append(f"{word}: {'✅' if in_clean else '❌ LOST'}")

    if preserved:
        print(f"\n  Key words preserved:")
        for p in preserved:
            print(f"    {p}")

    # Show side-by-side preview
    print(f"\n  RAW TEXT (first 200 chars):")
    print(f"  {raw_text[:200]}")
    print(f"\n  CLEANED TEXT (first 200 chars):")
    print(f"  {test_page['clean_text'][:200]}")


if __name__ == "__main__":
    quality_ok = validate_text_quality()
    compare_with_source_pdf()

    print("\n" + "=" * 70)
    if quality_ok:
        print("✅ VALIDATION PASSED - Ready for batch processing")
    else:
        print("⚠️ VALIDATION WARNINGS - Review before batch processing")
    print("=" * 70)
