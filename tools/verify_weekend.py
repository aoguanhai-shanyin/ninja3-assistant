"""Time-bounded real race acceptance; uses the normal worker control contract."""
import json
import argparse
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runner import RunControl
from weekend_race import run_weekend

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--practice',action='store_true')
    args = parser.parse_args()
    control = RunControl()
    path = Path('data/weekend-race')/('acceptance-'+time.strftime('%Y%m%d-%H%M%S')+'.jsonl')
    raced = False
    with path.open('w',encoding='utf-8') as log:
        def emit(kind,value):
            global raced
            log.write(json.dumps({'time':time.time(),'event':kind,'value':value},ensure_ascii=False)+'\n')
            log.flush()
            if kind in ('status','error','finished'):
                print(kind,value,flush=True)
            if kind == 'status' and value == 'C++ 滑步避障并释放获取的卡牌':
                raced = True
            if raced and kind == 'status' and value == '确认同屏竞速结算':
                control.request_stop()
        timer = threading.Timer(240,control.request_stop)
        timer.start()
        try:
            run_weekend('D:/Program Files/Netease/MuMu',1,control,emit,'25',record=False,practice=args.practice)
        finally:
            control.request_stop()
            timer.cancel()
