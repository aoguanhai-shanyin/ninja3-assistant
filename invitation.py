"""Small shared detector for the observed teammate invitation's reject button."""
from pathlib import Path
import sys
import cv2


class InvitationVision:
    def __init__(self):
        root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))/'assets'
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
