from config.settings import get_settings
from common.cache_manager import CacheManager
import random
st=get_settings(); cm=CacheManager(st)
paths=list(cm.rolling_dir.glob('*.feather'))
print('total rolling files:',len(paths))
random.shuffle(paths)
for p in paths[:1]:
    sym=p.stem
    df=cm.read(sym,'rolling')
    print('symbol',sym,'rows',0 if df is None else len(df))
    if df is None:
        continue
    print('columns:', list(df.columns))
