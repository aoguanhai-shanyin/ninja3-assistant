"""Daily shop inventory from the confirmed ninja-coin slots in one frame."""
import re

import cv2

from ninja3_assistant.activities.daily.vision import compact


def coin_stock(screen, vision):
    recovered = screen.ocr.get('daily_shop_stock',{})
    for y in (110,380):
        for x in (20,257,495,731):
            if vision.coin(screen,(x+35,y+200,x+215,y+250)):
                text = screen.text((x,y,x+230,y+250))
                count = re.search(r'剩余次数:(\d+)/',text)
                if count:
                    yield x,y,int(count[1])
                elif screen.text((x+10,y+30,x+220,y+76)) == '剩余次数:0':
                    # OCR can omit the denominator beside the sold-out zero.
                    yield x,y,0
                else:
                    yield x,y,recovered.get(f'{x}:{y}')


def recover_coin_stock(screen, vision, ocr, stop):
    slots = tuple(coin_stock(screen,vision))
    if any(count is not None and count > 0 for _,_,count in slots):
        return True
    for x,y,count in slots:
        if count is not None:
            continue
        # Full-frame OCR sometimes merges the adjacent 1/1 glyphs into a character.
        crop = screen.frame[y+30:y+76,x+10:x+220]
        enlarged = cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(enlarged,30,30,40,40,cv2.BORDER_CONSTANT,value=(0,0,0))
        result = ocr.read(image,stop=stop)
        if result is None:
            return False
        match = re.fullmatch(r'剩余次数:(\d+)/(\d+)',compact(result['text']))
        if match and int(match[1]) <= int(match[2]) and int(match[2]) > 0:
            count = int(match[1])
            screen.ocr.setdefault('daily_shop_stock',{})[f'{x}:{y}'] = count
            if count > 0:
                break
    return True
