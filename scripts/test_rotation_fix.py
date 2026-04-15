"""
Test rotation fix on NHS Grampian document.

Checks if previously empty rotated pages (113-120) now have content.
"""
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pdf.rotation_handler import is_rotated, get_rotation_metadata


def test_rotation_fix():
    """Test if rotation fix resolved empty page issues."""

    doc_id = "Grampian-2022-2023"
    output_dir = Path("data_processed") / doc_id

    print("=" * 70)
    print("ROTATION FIX VALIDATION")
    print("=" * 70)

    # Load pages
    pages_df = pd.read_parquet(output_dir / "pages.parquet")

    # Known rotated pages that were empty
    rotated_pages = [113, 114, 115, 117, 118, 119, 120]

    print(f"\n📊 Checking previously empty rotated pages...")
    print("-" * 70)

    fixed_count = 0
    still_empty = []

    for page_no in rotated_pages:
        page = pages_df[pages_df['page'] == page_no]

        if len(page) == 0:
            print(f"  ⚠️  Page {page_no}: Not found in output")
            continue

        page = page.iloc[0]
        text_len = len(page['clean_text'])
        rotation = page.get('rotation', 0)

        status = "✅ FIXED" if text_len >= 100 else "❌ Still empty"

        print(f"  Page {page_no}: {text_len:>5} chars, "
              f"rotation={rotation:>3}°, {status}")

        if text_len >= 100:
            fixed_count += 1
        else:
            still_empty.append(page_no)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Pages tested: {len(rotated_pages)}")
    print(f"  Fixed: {fixed_count}")
    print(f"  Still empty: {len(still_empty)}")

    if still_empty:
        print(f"  Empty pages: {still_empty}")

    # Check overall stats
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    total_pages = len(pages_df)
    empty_pages = len(pages_df[pages_df['clean_text'].str.len() < 50])

    print(f"  Total pages: {total_pages}")
    print(f"  Empty pages: {empty_pages} ({empty_pages / total_pages * 100:.1f}%)")

    # Expected: ~10-15 empty (structural), not 23
    if empty_pages < 15:
        print(f"  ✅ EXCELLENT: Empty page count reduced significantly")
    elif empty_pages < 20:
        print(f"  ✅ GOOD: Most rotated pages extracted")
    else:
        print(f"  ⚠️  Still high empty page count")

    # Sample text from page 113
    print("\n" + "=" * 70)
    print("SAMPLE: Page 113 (Previously Empty)")
    print("=" * 70)

    p113 = pages_df[pages_df['page'] == 113]
    if len(p113) > 0:
        text = p113.iloc[0]['clean_text']
        print(f"Length: {len(text)} chars")
        print(f"Preview:\n{text[:500]}")

    print("\n" + "=" * 70)

    if fixed_count >= len(rotated_pages) * 0.8:
        print("✅ ROTATION FIX SUCCESSFUL")
        return True
    else:
        print("⚠️  ROTATION FIX PARTIAL - Review settings")
        return False


if __name__ == "__main__":
    success = test_rotation_fix()
    sys.exit(0 if success else 1)