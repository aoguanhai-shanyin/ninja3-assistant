"""The player's house and the training continuation prompt."""

from dataclasses import dataclass, field
import re

import cv2

from ninja3_assistant.game_ui import GamePage, PageAction
from ninja3_assistant.vision.control_match import match_control, same_text, text_foreground
from ninja3_assistant.vision.game_update import game_update_banner


@dataclass(frozen=True)
class HouseView:
    training_remaining: int | None = None
    controls: dict[str, PageAction] = field(default_factory=dict)

    @property
    def can_collect_fragments(self):
        return 'claim_training' in self.controls

    @property
    def can_continue_training(self):
        return 'continue_training' in self.controls


class HousePage(GamePage):
    page_id = 'daily.house'
    related_pages = ('daily.house_continue',)

    @property
    def training_remaining(self):
        """Seconds remaining when this page was observed."""
        return self.observation.detail.training_remaining

    @property
    def can_collect_fragments(self):
        return self.observation.detail.can_collect_fragments

    @property
    def can_continue_training(self):
        return self.observation.detail.can_continue_training

    def collect_fragments(self):
        return self._perform('claim_training')

    def continue_training(self):
        if self.observation.page in self.related_pages:
            return self._perform('continue_training')
        current = self.ui.read_page(type(self))
        return current._perform('continue_training') if current is not None else None


def house_view(vision, screen):
    def control(name, point, verify):
        def guard(fresh):
            if game_update_banner(fresh):
                return False
            page,_ = vision.navigation(fresh)
            if screen.page == 'house':
                return page == 'house' and verify(fresh)
            return page in (None,'house','house_continue') and verify(fresh)

        return {name: PageAction(name, point, hold_seconds=.07, max_age=.25,
                                 guard=guard)}

    if screen.page == 'house_continue':
        prompt = (385,320,885,450)
        if '勾玉' in screen.text() or '是否消耗5饭团继续进行修炼' not in screen.text(prompt):
            return HouseView()
        amount = screen.hits(r'(?<!\d)5(?=饭团)',prompt)
        fee = screen.hits(r'饭团',prompt)
        training = screen.hits(r'继续进行修炼',prompt)
        confirm = screen.hits(r'确定',(700,475,920,575))
        if (len(amount) == len(fee) == len(training) == len(confirm) == 1
                and amount[0].box not in (fee[0].box,training[0].box)):
            labels = (amount[0],fee[0],training[0],confirm[0])
            return HouseView(controls=control('continue_training',confirm[0].point,
                lambda fresh: all(same_text(screen.frame,fresh,label.box) for label in labels)))
        return HouseView()
    if screen.page == 'house':
        countdown = clock_seconds(clock_words(screen.ocr))
        if countdown is not None and countdown > 0:
            return HouseView(training_remaining=countdown)
        template = vision.templates.get('house-complete')
        glyphs = text_foreground(template)

        def completion(frame):
            return match_control(frame,template,(270,670,380,711),mask=glyphs,threshold=.94,max_error=15)

        point = completion(screen.frame)
        if point:
            return HouseView(controls=control('claim_training',(320,650),
                lambda fresh: completion(fresh) == point))
        labels = screen.hits('修炼完成',(270,590,420,715))
        if len(labels) == 1:
            return HouseView(controls=control('claim_training',(320,650),
                lambda fresh: same_text(screen.frame,fresh,labels[0].box)))
        return HouseView(training_remaining=countdown)
    return HouseView()


def remaining_seconds(text):
    text = re.sub(r'\s+','',text).replace('時','时')
    text = re.sub(r'[小卜刁下][;；:：]?时','小时',text)
    text = re.sub(r'(?<=\d)[;；:：](?=分|秒)','',text)
    # An unreadable hour field cannot become a minutes-only deadline.
    if '时' in text and not re.search(r'(?<![0-9A-Za-z])\d+小时',text):
        return None
    total = 0
    matched = False
    for unit,multiplier in ((r'小时',3600),(r'分钟|分',60),(r'秒',1)):
        match = re.search(r'(?<![0-9A-Za-z])(\d+)(?:'+unit+r')',text)
        if match:
            total += int(match[1])*multiplier
            matched = True
        elif re.search(unit,text):
            return None
    return total if matched else None


def in_clock(word):
    x,y,w,h = word['bbox']
    return 250 <= x+w/2 < 410 and 675 <= y+h/2 < 710


def clock_words(result):
    return [word for word in result['words'] if in_clock(word)]


def clock_seconds(words):
    first_number = next((word for word in words if re.search(r'\d',word['text'])),None)
    # If the first number is in the minute field, OCR skipped the start of the clock.
    if first_number is None or first_number['bbox'][0] >= 310:
        return None
    return remaining_seconds(''.join(word['text'] for word in words))


def read_house_clock(frame, result, ocr, stop):
    words = result['words']
    if clock_seconds(clock_words(result)) is not None:
        return result
    clock_image = cv2.copyMakeBorder(cv2.resize(frame[675:710,250:410],None,fx=4,fy=4),
                                     30,30,30,30,cv2.BORDER_CONSTANT,value=(0,0,0))
    clock_result = ocr.read(clock_image, stop=stop)
    if clock_result is None:
        return None
    words = [w for w in words if not in_clock(w)]
    for word in clock_result['words']:
        x,y,w,h = word['bbox']
        words.append({**word,'bbox':[(x-30)/4+250,(y-30)/4+675,w/4,h/4]})
    return {**result, 'words':words, 'house_clock':clock_result}


def observe_house(vision, frame, ocr, stop):
    result = ocr.read(frame, stop=stop)
    if result is None:
        return None
    screen = vision.observe(frame,result)
    if screen.page != 'house':
        return screen
    result = read_house_clock(frame,result,ocr,stop)
    return vision.observe(frame,result) if result is not None else None
