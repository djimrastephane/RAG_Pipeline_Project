"""
Integration test for rotation handling.

Tests the full pipeline with rotation fix on a single document.
"""
from pathlib import Path
import subprocess
import sys


def run_integration_test():
    """Run full pipeline and validate results."""

    print("=" * 70)
    print("ROTATION FIX - INTEGRATION TEST")
    print("=" * 70)

    # Run preprocessing
    print("\n1. Running preprocessing pipeline...")
    print("-" * 70)

    result = subprocess.run(
        ['python', 'scripts/preprocess_hybrid.py'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Preprocessing failed:")
        print(result.stderr)
        return False

    print("✅ Preprocessing complete")

    # Run rotation validation
    print("\n2. Validating rotation fix...")
    print("-" * 70)

    result = subprocess.run(
        ['python', 'scripts/test_rotation_fix.py'],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print("⚠️  Rotation fix validation has warnings")
        return False

    print("✅ Rotation fix validated")

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print("✅ Pipeline ready for batch processing")

    return True


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)