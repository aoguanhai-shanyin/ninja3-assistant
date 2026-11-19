"""Keep imported quiz hints alongside answers verified by game feedback."""
from contextlib import closing
import json
import random
import re
import sqlite3
import unicodedata

from ninja3_assistant.activities.daily.resources import resource_path


def question_bank_path(folder, index):
    return resource_path(folder, index).with_name('landlady-quiz.sqlite3')


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).split()).strip('。！？?!')


def question_key(question, options):
    return normalize(question), json.dumps(sorted(map(normalize, options)), ensure_ascii=False)


_BRACKETS = str.maketrans('【〖〔(】〗〕)', '[[[[]]]]')
_TITLES = re.compile(r'(\[[^\[\]]+\])')


def _ocr_text(text):
    """Normalize observed OCR variants for lookup without changing stored keys."""
    text = normalize(text).translate(_BRACKETS)
    text = re.sub(r'(?<=[\u3400-\u9fff])[.・](?=[\u3400-\u9fff])', '·', text)
    text = re.sub(r'(?<=[0-9几])遒(?=闪电|$)', '道', text)
    return (text.replace('以下祁个', '以下哪个').replace('哪个忍木', '哪个忍术')
            .replace('是属干', '是属于'))


def _title_ocr_match(observed, expected):
    """Allow one observed glyph error inside a bracketed name; the rest must agree."""
    observed_parts, expected_parts = _TITLES.split(observed), _TITLES.split(expected)
    if len(observed_parts) != len(expected_parts):
        return False
    differences = [(i, got, want) for i, (got, want) in enumerate(zip(observed_parts, expected_parts))
                   if got != want]
    if len(differences) != 1 or not differences[0][0] % 2:
        return False
    _, got, want = differences[0]
    got, want = got[1:-1], want[1:-1]
    if len(got) < 3:
        return False
    # The recorded frames omit 诡 or read 光 as 先 in the named item.
    return any(got == want[:i] + replacement + want[i+1:]
               for i, char in enumerate(want)
               for expected_glyph, replacement in (('诡', ''), ('光', '先'))
               if char == expected_glyph)


def _ocr_hint(question, options, baseline):
    observed = _ocr_text(question)
    candidates = {}
    for text, answer in baseline:
        candidates.setdefault(_ocr_text(text), set()).add(_ocr_text(answer))
    answers = candidates.get(observed)
    if answers is None:
        near = [answers for text, answers in candidates.items() if _title_ocr_match(observed, text)]
        if len(near) != 1:
            return None
        answers = near[0]
    if len(answers) != 1:
        return None
    visible = [option for option in options if _ocr_text(option) in answers]
    return visible[0] if len(visible) == 1 else None


class QuizBank:
    def __init__(self, path):
        self.path = path

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute('''
                CREATE TABLE IF NOT EXISTS answers (
                    question TEXT NOT NULL,
                    options TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
                    PRIMARY KEY (question, options, answer)
                )
            ''')
            connection.execute('''
                CREATE TABLE IF NOT EXISTS baseline_answers (
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    PRIMARY KEY (question, answer)
                )
            ''')
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def choose(self, question, options):
        records = {}
        baseline = []
        key = question_key(question, options)
        if self.path.is_file():
            with closing(self.connect()) as connection:
                connection.execute('BEGIN')
                records = dict(connection.execute('''
                    SELECT answer, correct FROM answers WHERE question = ? AND options = ?
                ''', key))
                baseline = connection.execute('SELECT question, answer FROM baseline_answers').fetchall()
        confirmed = next((option for option in options if records.get(normalize(option)) == 1), None)
        if confirmed is not None:
            return confirmed
        exact = {answer for text, answer in baseline if text == key[0]}
        visible_hints = [option for option in options if normalize(option) in exact]
        if len(visible_hints) == 1 and normalize(visible_hints[0]) not in records:
            return visible_hints[0]
        hint = _ocr_hint(key[0], options, baseline) if baseline else None
        if hint is not None and normalize(hint) not in records:
            return hint
        untried = [option for option in options if normalize(option) not in records]
        return random.choice(untried) if untried else None

    def replace_baseline(self, entries):
        rows = set()
        for question, answer in entries:
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError('题库基准的题干和答案必须是文本')
            row = normalize(question), normalize(answer)
            if not all(row):
                raise ValueError('题库基准的题干和答案不能为空')
            rows.add(row)
        with closing(self.connect()) as connection, connection:
            connection.execute('DELETE FROM baseline_answers')
            connection.executemany('INSERT INTO baseline_answers VALUES (?, ?)', rows)
        return len(rows)

    def record(self, question, options, selected, *, correct, revealed=None):
        key = question_key(question, options)
        selected = normalize(selected)
        revealed = normalize(revealed) if revealed is not None else None
        right = selected if correct else revealed
        updates = [(*key, selected, int(correct))]
        if right is not None and right != selected:
            updates.append((*key, right, 1))
        with closing(self.connect()) as connection, connection:
            if right is not None:
                connection.execute('''
                    UPDATE answers SET correct = 0
                    WHERE question = ? AND options = ? AND correct = 1 AND answer != ?
                ''', (*key, right))
            connection.executemany('''
                INSERT INTO answers (question, options, answer, correct) VALUES (?, ?, ?, ?)
                ON CONFLICT (question, options, answer) DO UPDATE SET correct = excluded.correct
            ''', updates)
