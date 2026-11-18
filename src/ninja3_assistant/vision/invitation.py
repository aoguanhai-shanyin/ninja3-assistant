"""Shared detector for teammate invitations and their bounty details."""
from dataclasses import dataclass
import re

from ninja3_assistant.paths import resource_root
import cv2


@dataclass(frozen=True)
class BountyInvitation:
    source: str
    grade: str
    accept: tuple


class InvitationVision:
    def __init__(self):
        root = resource_root()/'assets'
        self.decline = cv2.imread(str(root/'invite-decline.png'), 0)
        self.accept = cv2.imread(str(root/'invite-accept.png'), 0)
        if self.decline is None or self.accept is None:
            raise RuntimeError('找不到队友邀请识别图片')

    def detect(self, gray):
        if gray.shape != (720, 1280):
            return None
        if gray.dtype != 'uint8':
            gray = gray.astype('uint8')
        if cv2.minMaxLoc(cv2.matchTemplate(gray[475:585,680:940], self.accept,
                                         cv2.TM_CCOEFF_NORMED))[1] < .92:
            return None
        score = cv2.matchTemplate(gray[475:585,340:610], self.decline, cv2.TM_CCOEFF_NORMED)
        _, maximum, _, at = cv2.minMaxLoc(score)
        if maximum >= .92:
            return (340+at[0]+self.decline.shape[1]//2, 475+at[1]+self.decline.shape[0]//2)
        return None

    def bounty_invitation(self, gray, ocr):
        """Read only the invitation's sender and title, never the background."""
        if self.detect(gray) is None:
            return None

        def text_in(box):
            x1,y1,x2,y2 = box
            return re.sub(r'\s+', '', ''.join(
                word['text'] for word in ocr.get('words', ())
                if x1 <= word['bbox'][0]+word['bbox'][2]/2 < x2
                and y1 <= word['bbox'][1]+word['bbox'][3]/2 < y2))

        sender = text_in((100,195,350,245))
        if '来自好友' in sender and '来自族员' not in sender:
            source = 'friend'
        elif '来自族员' in sender and '来自好友' not in sender:
            source = 'family'
        else:
            return None

        title = text_in((450,185,830,235)).upper()
        match = re.fullmatch(r'([DCBAS])级(?:每日|召令)悬?赏', title)
        if not match:
            return None
        score = cv2.matchTemplate(gray[475:585,680:940], self.accept, cv2.TM_CCOEFF_NORMED)
        _, maximum, _, at = cv2.minMaxLoc(score)
        if maximum < .92:
            return None
        return BountyInvitation(source,match[1],
            (680+at[0]+self.accept.shape[1]//2, 475+at[1]+self.accept.shape[0]//2))
