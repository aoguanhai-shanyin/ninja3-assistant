"""Summarize observed native timing events without mixing clock domains."""
import json
import math
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
boss, start, cycles = [], [], []
for path in sorted(root.glob('controller-part*.jsonl')):
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            item = json.loads(line)
            value = item.get('value')
            if not isinstance(value, dict):
                continue
            kind, data = value.get('event'), value.get('value')
            if kind == 'native_press' and data.get('kind') == 'boss' and data.get('capture_perf') is not None:
                boss.append(data['down_start_us']/1000-data['capture_perf']*1000)
            elif kind == 'native_timing' and data.get('kind') == 'start_down':
                start.append(data['capture_to_down_ms'])
            elif kind == 'native_touch' and isinstance(data, str) and data.startswith('CYCLE '):
                cycles.append(int(data.split()[2])/1000)
        except (ValueError, KeyError, TypeError, IndexError):
            continue
def summarize(values):
    if not values:
        return {'n':0}
    ordered = sorted(values)
    return {'n':len(values), 'p50_ms':statistics.median(values),
            'p95_ms':ordered[math.ceil(.95*len(ordered))-1],
            'max_ms':max(values)}
delta = [b-a for a,b in zip(cycles,cycles[1:]) if 0 < b-a < 500]
print(json.dumps({'boss_capture_request_to_down':summarize(boss),
                  'ground_capture_request_to_first_down':summarize(start),
                  'completed_cycle_start_interval':summarize(delta)}, indent=2))
