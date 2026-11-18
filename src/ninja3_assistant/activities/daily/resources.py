"""Per-instance house countdown used to reserve naturally regenerating food."""
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import sys

from ninja3_assistant.paths import project_root


def resource_path(folder, index):
    root = Path(sys.executable).parent if getattr(sys,'frozen',False) else project_root()/'data/runtime'
    key = hashlib.sha256((str(Path(folder).resolve()).casefold()+':'+str(index)).encode()).hexdigest()[:16]
    return root/'daily-cleanup'/('resources-'+key+'.json')


def remaining_seconds(text):
    text = re.sub(r'\s+','',text).replace('時','时')
    text = re.sub(r'[小卜刁][;；:：]?时','小时',text)
    # Do not turn an unreadable hour field into a minutes-only deadline.
    if '时' in text and not re.search(r'\d+小时',text):
        return None
    total = 0
    matched = False
    for unit,multiplier in ((r'小时',3600),(r'分钟|分',60),(r'秒',1)):
        match = re.search(r'(\d+)(?:'+unit+r')',text)
        if match:
            total += int(match[1])*multiplier
            matched = True
    return total if matched else None


def reserve_food(training_finishes_at, now):
    if training_finishes_at is None:
        return 5
    # Unknown regeneration phase: count complete 30-minute periods only.
    recoveries = max(0,int((training_finishes_at-now)//1800))
    return max(0,5-recoveries)


def load_deadline(path):
    try:
        value = float(json.loads(path.read_text(encoding='utf-8')).get('training_finishes_at'))
    except (AttributeError, OSError, OverflowError, TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def save_deadline(path, deadline, observed_at):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'training_finishes_at':deadline,'observed_at':observed_at,
        'food_per_training':5,'food_regeneration_seconds':1800},indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)
