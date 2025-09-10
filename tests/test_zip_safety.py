"""
Tests to ensure zip() calls use strict=True for safety.

Python 3.10+ introduced the strict parameter for zip() to catch mismatched
iterator lengths. Using strict=True prevents silent data corruption when
iterating over sequences of different lengths.
"""
# ALLOW_UNSAFE_ZIP_FOR_TESTING
import ast
import os
from pathlib import Path


def test_no_zip_calls_with_strict_false():
    """Ensure no zip() calls use strict=False in the codebase."""
    
    def check_file_for_unsafe_zip(file_path: Path) -> list[str]:
        """Check a Python file for zip() calls with strict=False."""
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
                    # Check if strict parameter is used and set to False
                    for keyword in node.keywords:
                        if keyword.arg == 'strict':
                            if (isinstance(keyword.value, ast.Constant) and 
                                keyword.value.value is False):
                                violations.append(
                                    f"{file_path}:{node.lineno}: zip() call with strict=False found"
                                )
                            # We could also check for NameConstant for older Python versions
                            elif (hasattr(ast, 'NameConstant') and 
                                  isinstance(keyword.value, ast.NameConstant) and 
                                  keyword.value.value is False):
                                violations.append(
                                    f"{file_path}:{node.lineno}: zip() call with strict=False found"
                                )
        except (SyntaxError, UnicodeDecodeError):
            # Skip files that can't be parsed or read
            pass
        return violations
    
    # Get the repository root
    repo_root = Path(__file__).parent.parent
    
    # Find all Python files
    violations = []
    for py_file in repo_root.rglob("*.py"):
        # Skip files in excluded directories
        if any(excluded in py_file.parts for excluded in ["data_cache", "results_csv", "logs", ".git"]):
            continue
        violations.extend(check_file_for_unsafe_zip(py_file))
    
    # Assert no violations found
    if violations:
        violation_msg = "\n".join(violations)
        raise AssertionError(
            f"Found zip() calls with strict=False. These should use strict=True for safety:\n{violation_msg}"
        )


def test_zip_strict_true_example():
    """Test that demonstrates proper zip() usage with strict=True."""
    # Example of safe zip usage
    list1 = [1, 2, 3]
    list2 = ['a', 'b', 'c']
    
    # This should work fine
    result = list(zip(list1, list2, strict=True))
    assert result == [(1, 'a'), (2, 'b'), (3, 'c')]
    
    # This should raise ValueError with mismatched lengths
    list3 = [1, 2, 3, 4]  # Different length
    try:
        list(zip(list1, list3, strict=True))
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass  # This is expected


def test_zip_strict_false_danger():
    """Test that demonstrates why strict=False is dangerous."""
    list1 = [1, 2, 3]
    list2 = ['a', 'b']  # Different length - shorter
    
    # With strict=False (or default), this silently truncates
    # NOTE: This is intentionally unsafe for demonstration purposes
    result_unsafe = list(zip(list1, list2, strict=False))  # Explicitly unsafe
    assert result_unsafe == [(1, 'a'), (2, 'b')]  # Data lost silently!
    
    # With strict=True, this would raise ValueError (safer)
    try:
        list(zip(list1, list2, strict=True))
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass  # This is the safe behavior we want