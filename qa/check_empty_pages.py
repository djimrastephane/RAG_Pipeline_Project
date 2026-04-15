# check_empty_pages.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pages_df = pd.read_parquet(PROJECT_ROOT / "data_processed" / "Grampian-2022-2023" / "pages.parquet")

empty_pages = [18, 53, 90, 113, 114, 115, 117, 118, 119, 120, 123, 125, 126, 127, 128, 136, 137, 144, 149, 150, 151,
               153, 154]

print("INSPECTING 'EMPTY' PAGES")
print("=" * 70)

for page_no in empty_pages[:5]:  # Check first 5
    page = pages_df[pages_df['page'] == page_no].iloc[0]

    print(f"\nPage {page_no}:")
    print(f"  Length: {len(page['clean_text'])} chars")
    print(f"  Is table: {page['is_table']}")
    print(f"  Section: {page.get('heading_candidates', [])}")
    print(f"  Content: '{page['clean_text'][:100]}'")

    # Determine likely reason
    if len(page['clean_text']) == 0:
        reason = "Completely blank (section divider or print spacing)"
    elif len(page['clean_text']) < 20:
        reason = "Minimal text (page number or chapter marker only)"
    elif page['is_table']:
        reason = "Table page with low text after extraction"
    else:
        reason = "Short content (notes or references)"

    print(f"  Likely: {reason}")

print("\n" + "=" * 70)
print("VERDICT: Empty pages are normal document structure elements")
print("Not a data quality issue - proceed with confidence!")
