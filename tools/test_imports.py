import importlib

try:
    r = importlib.import_module("scripts.run_all_systems_today")
    print("runner loaded")
    print("_PER_SYSTEM_EXIT_BULK" in globals())
except Exception as e:
    print("runner import error", e)

try:
    u = importlib.import_module("apps.app_today_signals")
    print("ui loaded")
except Exception as e:
    print("ui import error", e)
