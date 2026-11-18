"""Keep imported quiz hints alongside answers verified by game feedback."""
from contextlib import closing
import json
import random
import sqlite3
import unicodedata

from ninja3_assistant.activities.daily.resources import resource_path


def question_bank_path(folder, index):
    return resource_path(folder, index).with_name('landlady-quiz.sqlite3')


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).split()).strip('。！？?!')


def question_key(question, options):
    return normalize(question), json.dumps(sorted(map(normalize, options)), ensure_ascii=False)


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
        baseline = set()
        if self.path.is_file():
            key = question_key(question, options)
            with closing(self.connect()) as connection:
                connection.execute('BEGIN')
                records = dict(connection.execute('''
                    SELECT answer, correct FROM answers WHERE question = ? AND options = ?
                ''', key))
                baseline = {answer for answer, in connection.execute('''
                    SELECT answer FROM baseline_answers WHERE question = ?
                ''', (key[0],))}
        confirmed = next((option for option in options if records.get(normalize(option)) == 1), None)
        if confirmed is not None:
            return confirmed
        visible_hints = [option for option in options if normalize(option) in baseline]
        if len(visible_hints) == 1 and normalize(visible_hints[0]) not in records:
            return visible_hints[0]
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
