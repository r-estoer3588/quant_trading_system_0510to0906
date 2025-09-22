"""
Demonstration of the getattr redundancy fix mentioned in issue #41.

This module shows the correct and incorrect patterns for getattr usage
and provides a function to fix such patterns automatically.
"""

from __future__ import annotations

import re


def demonstrate_getattr_redundancy():
    """Demonstrate the redundancy issue and its fix."""

    class Position:
        def __init__(self, unrealized_plpc: float | None = None):
            if unrealized_plpc is not None:
                self.unrealized_plpc = unrealized_plpc

    # Test cases
    pos_with_value = Position(0.05)
    pos_without_attr = Position()

    print("=== getattr Redundancy Fix Demonstration ===\n")

    # Redundant pattern (from PR #37 review)
    print("❌ Redundant pattern:")
    print("   float(getattr(pos, 'unrealized_plpc', 0) or 0)")
    print("   ↳ The 'or 0' is unnecessary since getattr already defaults to 0\n")

    # Fixed pattern
    print("✅ Fixed pattern:")
    print("   float(getattr(pos, 'unrealized_plpc', 0))")
    print("   ↳ Cleaner and achieves the same result\n")

    # Test both patterns
    print("Testing with position that has unrealized_plpc = 0.05:")
    redundant_result = float(getattr(pos_with_value, "unrealized_plpc", 0) or 0)
    fixed_result = float(getattr(pos_with_value, "unrealized_plpc", 0))
    print(f"   Redundant pattern result: {redundant_result}")
    print(f"   Fixed pattern result:     {fixed_result}")
    print(f"   ✓ Both give the same result: {redundant_result == fixed_result}\n")

    print("Testing with position that lacks unrealized_plpc:")
    redundant_result = float(getattr(pos_without_attr, "unrealized_plpc", 0) or 0)
    fixed_result = float(getattr(pos_without_attr, "unrealized_plpc", 0))
    print(f"   Redundant pattern result: {redundant_result}")
    print(f"   Fixed pattern result:     {fixed_result}")
    print(f"   ✓ Both give the same result: {redundant_result == fixed_result}\n")


def fix_getattr_redundancy(code: str) -> tuple[str, int]:
    """
    Fix redundant 'or 0' patterns in getattr calls.

    Args:
        code: Python source code as string

    Returns:
        tuple of (fixed_code, number_of_fixes)
    """

    patterns = [
        # float(getattr(..., 0) or 0) -> float(getattr(..., 0))
        (
            r"float\(\s*getattr\([^,]+,\s*[^,]+,\s*0(?:\.0)?\)\s*or\s*0(?:\.0)?\s*\)",
            lambda m: m.group(0).replace(" or 0)", ")").replace(" or 0.0)", ")"),
        ),
        # int(getattr(..., 0) or 0) -> int(getattr(..., 0))
        (
            r"int\(\s*getattr\([^,]+,\s*[^,]+,\s*0\)\s*or\s*0\s*\)",
            lambda m: m.group(0).replace(" or 0)", ")"),
        ),
        # getattr(..., 0) or 0 -> getattr(..., 0)
        (
            r"getattr\([^,]+,\s*[^,]+,\s*0(?:\.0)?\)\s*or\s*0(?:\.0)?",
            lambda m: m.group(0).split(" or ")[0],
        ),
    ]

    fixed_code = code
    total_fixes = 0

    for pattern, replacement in patterns:
        matches = re.findall(pattern, fixed_code)
        total_fixes += len(matches)
        fixed_code = re.sub(pattern, replacement, fixed_code)

    return fixed_code, total_fixes


if __name__ == "__main__":
    demonstrate_getattr_redundancy()

    # Test the fix function
    test_code = """
    # Examples of redundant patterns
    value1 = float(getattr(pos, "unrealized_plpc", 0) or 0)
    value2 = int(getattr(obj, "count", 0) or 0)
    value3 = getattr(item, "score", 0.0) or 0.0
    """

    print("=== Automatic Fix Demonstration ===\n")
    print("Original code:")
    print(test_code)

    fixed, num_fixes = fix_getattr_redundancy(test_code)
    print(f"\nFixed code ({num_fixes} patterns fixed):")
    print(fixed)
