import queue
import unittest
from types import SimpleNamespace
from app import Application

class Label:
    def __init__(self): self.text = ''
    def set(self, value): self.text = value

class AppCounterTest(unittest.TestCase):
    def test_counter_remains_visible_across_stopping_and_restart(self):
        app = object.__new__(Application)
        app.events = queue.Queue()
        app.kill_count = 0
        app.kills = Label()
        app.root = SimpleNamespace(after=lambda *args: None)
        app.start_button = SimpleNamespace(configure=lambda **kwargs: None)
        app.events.put(('kill', 1))
        app.events.put(('finished', ''))
        app.poll()
        self.assertEqual(app.kills.text, '本次成功击杀：1 次')
        app.events.put(('kill', 1))
        app.poll()
        self.assertEqual(app.kills.text, '本次成功击杀：2 次')

if __name__ == '__main__':
    unittest.main()
