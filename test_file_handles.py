import sys
from pathlib import Path

import streamlit as st

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 修正されたモジュールをテスト
try:
    from common.structured_logging import MetricsCollector, TradingSystemLogger

    st.title("File Handle Leak Test")

    if st.button("Test Multiple MetricsCollector Instances"):
        collectors = [MetricsCollector() for i in range(20)]
        st.success(f"✅ Created {len(collectors)} MetricsCollector instances successfully!")

    if st.button("Test TradingSystemLogger"):
        logger = TradingSystemLogger()
        st.success("✅ TradingSystemLogger created successfully!")

    if st.button("Test Combined"):
        for i in range(10):
            MetricsCollector()
            if i % 3 == 0:
                TradingSystemLogger()
        st.success("✅ Combined test completed successfully!")

    st.info("All tests completed without file handle errors.")

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
