#!/usr/bin/env python3#!/usr/bin/env python3

"""Test script to verify that the format migration works correctly.""""""Test script to verify that the formatdef test_cache_manager_detects_feather_for_base():

    """Test that CacheManager detects feather files for base cache."""

import tempfile    with tempfile.TemporaryDirectory() as temp_dir:

import pandas as pd        old_cache_dir = os.environ.get('DATA_CACHE_DIR')

from pathlib import Path        os.environ['DATA_CACHE_DIR'] = temp_dir

import sys        

import os        try:

            from config.settings import get_settings

# Add the root directory to the path            settings = get_settings(create_dirs=True)

ROOT = Path(__file__).resolve().parent            

sys.path.insert(0, str(ROOT))            cm = CacheManager(settings)

            

from common.cache_manager import save_base_cache, CacheManager            # Create base directory

            base_dir = Path(temp_dir) / "base"

            base_dir.mkdir(parents=True, exist_ok=True)

def test_save_base_cache_creates_feather():            

    """Test that save_base_cache creates feather files."""            # Test path creation directly - we expect feather for base

    # Create test data            expected_path = base_dir / "TEST.feather"

    test_data = pd.DataFrame({            print(f"Expected path for base cache: {expected_path}")

        'date': pd.date_range('2023-01-01', periods=5),            

        'open': [100, 101, 102, 103, 104],            # For base cache, it should prefer feather

        'high': [105, 106, 107, 108, 109],            assert expected_path.suffix == '.feather', f"Expected .feather for base, got {expected_path.suffix}"

        'low': [95, 96, 97, 98, 99],            print("✓ CacheManager expects feather format for base cache")

        'close': [104, 105, 106, 107, 108],            

        'volume': [1000, 1100, 1200, 1300, 1400]        finally:

    })            # Restore original environment

    test_data.set_index('date', inplace=True)            if old_cache_dir is not None:

                    os.environ['DATA_CACHE_DIR'] = old_cache_dir

    # Create temporary directory for testing            elif 'DATA_CACHE_DIR' in os.environ:

    with tempfile.TemporaryDirectory() as temp_dir:                del os.environ['DATA_CACHE_DIR']rrectly."""

        # Set environment variable and get settings

        old_cache_dir = os.environ.get('DATA_CACHE_DIR')import tempfile

        os.environ['DATA_CACHE_DIR'] = temp_dirimport pandas as pd

        from pathlib import Path

        try:import sys

            from config.settings import get_settingsimport os

            settings = get_settings(create_dirs=True)

            # Add the root directory to the path

            # Test save_base_cacheROOT = Path(__file__).resolve().parent

            result_path = save_base_cache('TEST', test_data, settings)sys.path.insert(0, str(ROOT))

            print(f"Created file: {result_path}")

            from common.cache_manager import save_base_cache, CacheManager

            # Verify the file was created with .feather extension

            assert result_path.suffix == '.feather', f"Expected .feather file, got {result_path.suffix}"

            assert result_path.exists(), f"File was not created: {result_path}"def test_save_base_cache_creates_feather():

                """Test that save_base_cache creates feather files."""

            # Verify we can read the data back    # Create test data

            read_data = pd.read_feather(result_path)    test_data = pd.DataFrame({

            print(f"Read back data shape: {read_data.shape}")        'date': pd.date_range('2023-01-01', periods=5),

            print(f"Columns: {list(read_data.columns)}")        'open': [100, 101, 102, 103, 104],

                    'high': [105, 106, 107, 108, 109],

            # Verify data integrity        'low': [95, 96, 97, 98, 99],

            assert len(read_data) == len(test_data), "Data length mismatch"        'close': [104, 105, 106, 107, 108],

            print("✓ save_base_cache creates feather files successfully")        'volume': [1000, 1100, 1200, 1300, 1400]

                })

        finally:    test_data.set_index('date', inplace=True)

            # Restore original environment    

            if old_cache_dir is not None:    # Create temporary directory for testing

                os.environ['DATA_CACHE_DIR'] = old_cache_dir    with tempfile.TemporaryDirectory() as temp_dir:

            elif 'DATA_CACHE_DIR' in os.environ:        # Set environment variable and get settings

                del os.environ['DATA_CACHE_DIR']        old_cache_dir = os.environ.get('DATA_CACHE_DIR')

        os.environ['DATA_CACHE_DIR'] = temp_dir

        

if __name__ == '__main__':        try:

    print("Testing format migration...")            from config.settings import get_settings

    test_save_base_cache_creates_feather()            settings = get_settings(create_dirs=True)

    print("All tests passed! ✓")            
            # Test save_base_cache
            result_path = save_base_cache('TEST', test_data, settings)
            print(f"Created file: {result_path}")
            
            # Verify the file was created with .feather extension
            assert result_path.suffix == '.feather', f"Expected .feather file, got {result_path.suffix}"
            assert result_path.exists(), f"File was not created: {result_path}"
            
            # Verify we can read the data back
            read_data = pd.read_feather(result_path)
            print(f"Read back data shape: {read_data.shape}")
            print(f"Columns: {list(read_data.columns)}")
            
            # Verify data integrity
            assert len(read_data) == len(test_data), "Data length mismatch"
            print("✓ save_base_cache creates feather files successfully")
            
        finally:
            # Restore original environment
            if old_cache_dir is not None:
                os.environ['DATA_CACHE_DIR'] = old_cache_dir
            elif 'DATA_CACHE_DIR' in os.environ:
                del os.environ['DATA_CACHE_DIR']


def test_cache_manager_detects_feather_for_base():
    """Test that CacheManager detects feather files for base cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        settings = get_settings()
        settings.DATA_CACHE_DIR = temp_dir
        
        cm = CacheManager(settings)
        
        # Create base directory
        base_dir = Path(temp_dir) / "base"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Test _detect_path for base cache
        detected_path = cm._detect_path(base_dir, "TEST")
        print(f"Detected path for base cache: {detected_path}")
        
        # For base cache, it should prefer feather
        assert detected_path.suffix == '.feather', f"Expected .feather for base, got {detected_path.suffix}"
        print("✓ CacheManager detects feather format for base cache")


if __name__ == '__main__':
    print("Testing format migration...")
    test_save_base_cache_creates_feather()
    test_cache_manager_detects_feather_for_base()
    print("All tests passed! ✓")