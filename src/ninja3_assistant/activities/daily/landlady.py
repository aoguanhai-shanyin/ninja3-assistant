"""Learn the landlady quiz one confirmed answer at a time."""
from dataclasses import dataclass
import re

import cv2
import numpy as np

from ninja3_assistant.activities.daily.arena import BACK, QUIZ_ENTRY, TOP_LEFT, lobby_entry, navigate, tap
from ninja3_assistant.activities.daily.quiz_bank import QuizBank, normalize, question_bank_path
from ninja3_assistant.activities.daily.vision import DailyScreen


QUESTION_BOX = (435, 112, 1200, 297)
ROW_CENTERS = (370, 450, 530)
FINISHED = r'今[天夭].*(?:已经?答完|全部答完)|明[天夭]再(?:来|答)|今日.*(?:已完成|已答完)|今日已领取'
CLAIM = r'(?<!已)领取奖励'
QUIZ_GUARD = (640, 335, 1180, 563)


def prompt_visible(frame):
    bubble = frame[95:109, 425:1215]
    return (bubble.min(axis=2) > 225).mean() > .9


def question_panel_visible(frame):
    border = frame[322:335, 655:1165]
    return prompt_visible(frame) and (border.min(axis=2) > 225).mean() > .9


def observe(screen):
    title = rf'{QUIZ_ENTRY}|[板极版]娘[问問向]答'
    completion = screen.frame[212:228, 425:1215]
    if screen.hits(title, TOP_LEFT) and (prompt_visible(screen.frame) or (completion.min(axis=2) > 225).mean() > .9):
        return 'landlady-quiz', BACK
    return None


@dataclass(frozen=True)
class Question:
    number: int | None
    text: str
    options: dict


def read_question(screen):
    # OCR merges consecutive Chinese and Latin glyphs, so use each option's own row.
    options = {}
    for label, y in zip('ABC', ROW_CENTERS):
        row = (645, y - 33, 1179, y + 33)
        match = re.fullmatch(
            rf'[{label}{label.lower()}](?:[.．、:：)）·,，]|(?=[\u3400-\u4dbf\u4e00-\u9fff]))(.+)',
            screen.text(row),
        )
        if match is None:
            return None
        option = normalize(match[1])
        if not option or option in options:
            return None
        options[option] = (910, y)

    prompt = question_prompt(screen)
    if prompt is None:
        return None
    progress, text = prompt
    return Question(progress, text, options)


def question_prompt(screen):
    text = screen.text(QUESTION_BOX)
    number = re.match(r'第(10|[0-9])?(?:[/／]10)?[iI丨]?[问問向司][:：]?', text)
    if number is None or len(text[number.end():]) < 4:
        return None
    progress = int(number[1]) if number[1] not in (None, '0') else None
    return progress, text[number.end():]


def same_prompt(before, fresh):
    before = before[115:282, 455:1180].astype(np.int16)
    fresh = fresh[115:282, 455:1180].astype(np.int16)
    return np.abs(before - fresh).mean() < .35


def recover_question_ocr(frame, result, ocr, stop):
    # The large scene can cause Windows OCR to omit a short answer such as a single character.
    original_screen = DailyScreen(frame, result)
    if not question_panel_visible(frame) or read_question(original_screen) or any(
            row is not None for row in feedback(frame)):
        return result
    x1, y1, x2, y2 = (445, 112, 1190, 570)
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    _, contrasted = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    local = ocr.read(cv2.cvtColor(contrasted, cv2.COLOR_GRAY2RGB), stop=stop)
    if local is None:
        return result
    text_start = 330 if question_prompt(original_screen) else y1
    words = [word for word in result['words'] if not (
        435 <= word['bbox'][0] + word['bbox'][2] / 2 < 1200 and
        text_start <= word['bbox'][1] + word['bbox'][3] / 2 < y2)]
    for word in local['words']:
        x, y, width, height = word['bbox']
        if y + y1 + height / 2 >= text_start:
            words.append({**word, 'bbox': (x + x1, y + y1, width, height)})
    improved = {**result, 'words': words, 'quiz_text': local['text']}
    return improved if read_question(DailyScreen(frame, improved)) else result


def feedback(frame):
    """Read the selected green/orange row and the revealed green check mark."""
    if not question_panel_visible(frame):
        return None, None, None
    correct = wrong = revealed = None
    for row, y in enumerate(ROW_CENTERS):
        patch = frame[y - 18:y + 18, 1118:1142].astype(np.int16)
        red, green, blue = patch.transpose(2, 0, 1)
        if ((green >= 177) & (green - red > 30) & (green - blue > 75)).mean() > .85:
            correct = row
        if ((red >= 220) & (red - green > 100) & (green - blue > 40)).mean() > .85:
            wrong = row
        icon = frame[y - 27:y + 27, 681:741].astype(np.int16)
        red, green, blue = icon.transpose(2, 0, 1)
        if ((green - red > 28) & (green - blue > 32)).mean() > .1:
            revealed = row
    return correct, wrong, revealed


class LandladyQuiz:
    def __init__(self, path):
        self.bank = QuizBank(path)
        self.pending = None
        self.finished = False
        self.unavailable = False
        self.reward_pending = False

    def step(self, screen, vision):
        from ninja3_assistant.activities.daily.flow import DailyDecision

        if self.finished:
            if screen.page == 'home':
                return DailyDecision('unavailable' if self.unavailable else 'done', status=(
                    '老板娘问答无未尝试选项，保留未完成状态' if self.unavailable else '老板娘今日问答完成'))
            if screen.page in ('landlady-quiz', 'arena', 'arena-menu'):
                return tap(screen.point, '返回主页，继续每日清理')
        if screen.page == 'arena':
            entry = lobby_entry(screen)
            return tap(entry, '进入老板娘问答') if entry else DailyDecision(status='寻找老板娘问答入口')
        if screen.page != 'landlady-quiz':
            if screen.page == 'home':
                self.pending = None
            return navigate(screen, vision)

        if self.observe_feedback(screen.frame):
            return DailyDecision(status='等待老板娘展示下一题')

        question = read_question(screen)
        if question is None:
            text = screen.text((435, 112, 1200, 370))
            claim = screen.hits(CLAIM, (755, 380, 1235, 530))
            if len(claim) == 1 and not self.reward_pending:
                return tap(claim[0].point, '领取老板娘答题奖励')
            if re.search(FINISHED, text):
                self.finished = True
                return tap(BACK, '返回主页，继续每日清理')
            start = screen.hits(r'准备好了[吗嗎]?|继续答题|开始答题', (755, 380, 1235, 530))
            if len(start) == 1:
                return tap(start[0].point, '开始老板娘每日问答')
            return DailyDecision(status='等待识别完整题干和选项',
                                 defer_on_timeout=question_panel_visible(screen.frame))
        if self.pending:
            previous, _, before_options = self.pending
            options = screen.frame[335:563, 640:1180].astype(np.int16)
            changed = np.abs(options - before_options).mean() > 3
            if changed and (question.text != previous.text or question.options != previous.options):
                # A new question is visible, but a missed result never changes the bank.
                self.pending = None
            else:
                return DailyDecision(status='等待老板娘确认答题结果')
        selected = self.bank.choose(question.text, question.options)
        if selected is None:
            skip = screen.hits('跳过(?:本题)?', (405, 340, 1240, 700))
            if len(skip) == 1:
                return tap(skip[0].point, '跳过所有选项均已确认错误的题目')
            self.finished = self.unavailable = True
            return tap(BACK, '返回主页，继续每日清理')
        return tap(question.options[selected], '提交老板娘答题选项', QUIZ_GUARD)

    def observe_feedback(self, frame):
        correct, wrong, revealed = feedback(frame)
        if self.pending and (correct is not None or wrong is not None):
            question, selected, _ = self.pending
            options = tuple(question.options)
            selected_row = options.index(selected)
            if selected_row in (correct, wrong) and (correct is None or wrong is None):
                answer = options[revealed] if revealed is not None and revealed != selected_row else None
                self.bank.record(question.text, options, selected, correct=correct == selected_row, revealed=answer)
                self.pending = None
        return correct is not None or wrong is not None

    def completed(self, decision, screen):
        if decision.status == '提交老板娘答题选项':
            question = read_question(screen)
            selected = next(option for option, point in question.options.items() if point == decision.point)
            self.pending = question, selected, screen.frame[335:563, 640:1180].astype(np.int16)
        elif decision.status == '领取老板娘答题奖励':
            self.reward_pending = True
