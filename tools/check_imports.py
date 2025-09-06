import importlib

mods = [
    'core.system1',
    'strategies.system1_strategy',
    'common.ui_components',
]

for m in mods:
    try:
        importlib.import_module(m)
        print('OK', m)
    except Exception as e:
        print('ERR', m, repr(e))

