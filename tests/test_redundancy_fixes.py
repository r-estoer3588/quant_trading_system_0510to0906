"""Test for the redundancy fixes applied in issue #41."""

from __future__ import annotations

from tools.fix_getattr_redundancy import fix_getattr_redundancy


def test_getattr_redundancy_fix():
    """Test that getattr redundancy patterns are correctly identified and fixed."""
    
    # Test the exact pattern mentioned in the issue
    test_code = 'float(getattr(pos, "unrealized_plpc", 0) or 0)'
    fixed_code, num_fixes = fix_getattr_redundancy(test_code)
    
    assert num_fixes == 1
    assert fixed_code == 'float(getattr(pos, "unrealized_plpc", 0))'


def test_getattr_patterns_behavior():
    """Test that fixed patterns behave identically to original redundant patterns."""
    
    class DummyPos:
        def __init__(self, unrealized_plpc=None):
            if unrealized_plpc is not None:
                self.unrealized_plpc = unrealized_plpc
    
    # Test with value present
    pos1 = DummyPos(0.05)
    
    # Both patterns should give the same result
    redundant_result = float(getattr(pos1, "unrealized_plpc", 0) or 0)
    fixed_result = float(getattr(pos1, "unrealized_plpc", 0))
    
    assert redundant_result == fixed_result == 0.05
    
    # Test with value missing
    pos2 = DummyPos()
    
    redundant_result = float(getattr(pos2, "unrealized_plpc", 0) or 0)
    fixed_result = float(getattr(pos2, "unrealized_plpc", 0))
    
    assert redundant_result == fixed_result == 0.0


def test_dict_get_fixes():
    """Test that the dict.get fixes work correctly."""
    
    # Test the notifier.py fix
    def test_nested_get_pattern():
        t = {"qty": 5}
        # Fixed pattern: no more redundant 'or 0'
        result = int(t.get("qty", t.get("shares", 0)))
        assert result == 5
        
        # When both keys are missing
        t2 = {}
        result2 = int(t2.get("qty", t2.get("shares", 0)))
        assert result2 == 0
        
        # When fallback key exists
        t3 = {"shares": 3}
        result3 = int(t3.get("qty", t3.get("shares", 0)))
        assert result3 == 3
    
    # Test the run_all_systems_today.py fix
    def test_score_pattern():
        r = {"score": 0.5}
        # Fixed pattern: use default in .get() instead of 'or'
        result = float(r.get("score", 0.0))
        assert result == 0.5
        
        # When key doesn't exist
        r2 = {}
        result2 = float(r2.get("score", 0.0))
        assert result2 == 0.0
    
    test_nested_get_pattern()
    test_score_pattern()


def test_multiple_getattr_patterns():
    """Test that multiple patterns can be fixed in one pass."""
    
    test_code = """
    value1 = float(getattr(pos, "unrealized_plpc", 0) or 0)
    value2 = int(getattr(obj, "count", 0) or 0)
    value3 = getattr(item, "score", 0.0) or 0.0
    """
    
    fixed_code, num_fixes = fix_getattr_redundancy(test_code)
    
    assert num_fixes == 3
    assert "or 0)" not in fixed_code
    assert "or 0.0" not in fixed_code
    assert 'float(getattr(pos, "unrealized_plpc", 0))' in fixed_code
    assert 'int(getattr(obj, "count", 0))' in fixed_code
    assert 'getattr(item, "score", 0.0)' in fixed_code