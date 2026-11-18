"""Import a UTF-8 quiz baseline without replacing game-verified answers."""

import argparse
import csv
from pathlib import Path
import sqlite3

from ninja3_assistant.activities.daily.quiz_bank import QuizBank
from ninja3_assistant.paths import project_root


def read_entries(path):
    entries = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ['题目', '答案']:
            raise ValueError('CSV 表头应为「题目,答案」')
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            if len(row) != 2 or not all(cell.strip() for cell in row):
                raise ValueError(f'第 {reader.line_num} 行需要完整的题目和答案')
            entries.append(tuple(row))
    if not entries:
        raise ValueError('CSV 中没有可导入的题目')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='UTF-8 CSV，表头为「题目,答案」')
    args = parser.parse_args()
    path = project_root() / 'data/runtime/daily-cleanup/landlady-quiz.sqlite3'
    try:
        count = QuizBank(path).replace_baseline(read_entries(args.source))
    except (OSError, ValueError, csv.Error, sqlite3.Error) as error:
        parser.error(str(error))
    print(f'已导入 {count} 条基准题目与答案')


if __name__ == '__main__':
    main()
