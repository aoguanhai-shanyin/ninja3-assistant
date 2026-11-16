"""Bounded local evidence for unreadable card slots; never guesses identities."""
import hashlib
import json
import time

import cv2

from ninja3_assistant.activities.intelligence.battle_vision import BOARD_BOXES, SHOP_BOXES


class UnknownCardRecorder:
    def __init__(self, output):
        self.output = output/'unknown-cards'
        self.seen = set()
        self.counts = {}

    def save(self, frame, battle):
        if battle is None or battle.kind != 'placement':
            return None
        slots = []
        for area,indices,boxes in (('board',battle.unknown_board,BOARD_BOXES),
                                   ('shop',battle.unknown_shop,SHOP_BOXES)):
            for index in indices:
                x1,y1,x2,y2 = boxes[index]
                crop = frame[y1:y2,x1:x2]
                thumbnail = cv2.resize(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),(12,16))//32
                key = (battle.round_number,area,index)
                fingerprint = (*key,hashlib.sha256(thumbnail.tobytes()).hexdigest())
                # Up to three visual variants per slot/round, 128 per run.
                if fingerprint in self.seen or self.counts.get(key,0) >= 3 or len(self.seen) >= 128:
                    continue
                self.output.mkdir(parents=True,exist_ok=True)
                name = f'{len(self.seen):04d}-round{battle.round_number}-{area}{index}'
                path = self.output/(name+'.png')
                if not cv2.imwrite(str(path),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)):
                    continue
                self.seen.add(fingerprint)
                self.counts[key] = self.counts.get(key,0)+1
                slots.append({'area':area,'slot':index,'box':[x1,y1,x2,y2],
                              'crop':str(path.relative_to(self.output.parent)),
                              'candidates':list(battle.card_candidates.get(f'{area}:{index}',()))})
        if not slots:
            return None
        source = self.output/f'frame-{len(self.seen):04d}.png'
        saved = cv2.imwrite(str(source),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        record = {'event':'unknown_cards','time':time.time(),'round':battle.round_number,
                  'coins':battle.coins,'selected_shop':battle.selected_shop,
                  'frame':str(source.relative_to(self.output.parent)) if saved else None,'slots':slots}
        with (self.output/'index.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        return record
