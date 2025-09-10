#!/usr/bin/env python3
"""
Custom linting script to check for unsafe zip() usage.

This script ensures that all zip() calls use strict=True to prevent
silent data corruption from mismatched iterator lengths.
"""
import ast
import sys
from pathlib import Path


def check_file_for_unsafe_zip(file_path: Path) -> list[str]:
    """Check a Python file for zip() calls with strict=False or missing strict parameter."""
    violations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Skip files that explicitly allow unsafe zip for testing/demonstration
        if "# ALLOW_UNSAFE_ZIP_FOR_TESTING" in content:
            return violations
            
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'zip':
                strict_found = False
                strict_is_true = False
                
                # Check if strict parameter is used
                for keyword in node.keywords:
                    if keyword.arg == 'strict':
                        strict_found = True
                        # Check if it's True
                        if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            strict_is_true = True
                        elif (hasattr(ast, 'NameConstant') and 
                              isinstance(keyword.value, ast.NameConstant) and 
                              keyword.value.value is True):
                            strict_is_true = True
                        break
                
                # Report violations
                if not strict_found:
                    violations.append(
                        f"{file_path}:{node.lineno}: zip() call missing strict=True parameter"
                    )
                elif not strict_is_true:
                    violations.append(
                        f"{file_path}:{node.lineno}: zip() call should use strict=True, not strict=False"
                    )
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed or read
        pass
    return violations


def main():
    """Main function to check all Python files for unsafe zip() usage."""
    if len(sys.argv) > 1:
        # Check specific files passed as arguments
        files_to_check = [Path(f) for f in sys.argv[1:] if f.endswith('.py')]
    else:
        # Check all Python files in the repository
        repo_root = Path(__file__).parent.parent
        files_to_check = list(repo_root.rglob("*.py"))
    
    all_violations = []
    for py_file in files_to_check:
        # Skip files in excluded directories
        if any(excluded in py_file.parts for excluded in ["data_cache", "results_csv", "logs", ".git"]):
            continue
        violations = check_file_for_unsafe_zip(py_file)
        all_violations.extend(violations)
    
    if all_violations:
        print("❌ Found unsafe zip() usage:")
        for violation in all_violations:
            print(f"  {violation}")
        print("\n💡 Fix: Use zip(..., strict=True) to prevent silent data corruption")
        sys.exit(1)
    else:
        print("✅ All zip() calls are safe (using strict=True)")
        sys.exit(0)


if __name__ == "__main__":
    main()