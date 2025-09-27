from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルート（apps/systems/ から2階層上）をパスに追加
sys.path.insert(0, str(Path(__file__).parents[2]))

# System7 UI のメインページ
import streamlit as st


def main():
    st.title("System7 - UI")
    st.write("System7 の UI ページです")


if __name__ == "__main__":
    main()
