from __future__ import annotations

from pathlib import Path
import sys

# プロジェクトルート（apps/systems/ から2階層上）をパスに追加
sys.path.insert(0, str(Path(__file__).parents[2]))

# System6 UI のメインページ
import streamlit as st


def main():
    st.title("System6 - UI")
    st.write("System6 の UI ページです")


if __name__ == "__main__":
    main()
