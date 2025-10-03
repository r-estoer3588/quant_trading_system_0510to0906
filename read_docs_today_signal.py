import os


def read_docs():
    doc_dir = "c:/Repos/quant_trading_system/docs/today_signal_scan"
    os.chdir(doc_dir)

    files = [f for f in os.listdir(".") if f.endswith(".md")]

    for filename in sorted(files):
        print(f"\n{'='*60}")
        print(f"📄 {filename}")
        print("=" * 60)

        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                print(content)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

        print()


if __name__ == "__main__":
    read_docs()
