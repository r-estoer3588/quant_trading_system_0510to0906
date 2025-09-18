from .server import main

if __name__ == "__main__":
    import anyio

    anyio.run(main)
