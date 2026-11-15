import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from mumu import GAME
from mumu_macro import GroundMacroGate, PRESS_ACTIONS, SCHEME, SkateMacro, selected_macro
from runner import RunControl


class MacroTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / (GAME + '-' + SCHEME + '.json')
        self.meta = self.root / (GAME + '.json')
        self.key = {'device': 'keyboard', 'scan_code': 66, 'text': 'F8', 'virtual_key': 119}
        self.item = {'type': 'Macro', 'key': self.key, 'press_actions': PRESS_ACTIONS[:], 'release_actions': [],
                     'editor_icon_scale': 1, 'icon': {}, 'rel_work_position': {}}
        self.write_config([self.item])
        self.meta.write_text(json.dumps({'Current': {SCHEME: {'configType': 'Customer', 'path': str(self.path)}}}),
                             encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, items):
        self.path.write_text(json.dumps({'keymaps': items}), encoding='utf-8')

    def test_only_exact_selected_unique_macro_is_allowed(self):
        self.assertTrue(selected_macro(self.root))
        self.write_config([dict(self.item, press_actions=['tap'])])
        self.assertFalse(selected_macro(self.root))
        self.write_config([dict(self.item, press_lua_chunk_info={'lua_script': 'unexpected'})])
        self.assertFalse(selected_macro(self.root))
        self.write_config([self.item, {'type': 'Click', 'key': self.key}])
        self.assertFalse(selected_macro(self.root))
        self.write_config([self.item])
        self.meta.write_text(json.dumps({'Current': {'other': {'path': str(self.path)}}}), encoding='utf-8')
        self.assertFalse(selected_macro(self.root))

    def test_lease_stop_and_scheme_change_release_held_key(self):
        messages = []
        stop = threading.Event()
        macro = SkateMacro(1, self.root, stop, lambda *args: messages.append(args) or True,
                           lambda *_: None, lease=.05)
        try:
            self.assertTrue(macro.renew())
            time.sleep(.09)
            self.assertEqual([m[1] for m in messages], [0x100, 0x101])
            self.assertTrue(macro.renew())
            self.write_config([dict(self.item, press_actions=['tap'])])
            self.assertFalse(macro.renew())
            self.assertEqual([m[1] for m in messages], [0x100, 0x101, 0x100, 0x101])
        finally:
            macro.close()

    def test_pause_and_stop_release_before_acknowledgement(self):
        messages = []
        control = RunControl()
        macro = SkateMacro(1, self.root, control.stop, lambda *args: messages.append(args) or True,
                           lambda *_: None)
        control.set_release_input(macro.release)
        try:
            self.assertTrue(control.run_if_active(macro.renew))
            self.assertTrue(control.pause())
            self.assertEqual(messages[-1][1], 0x101)
            self.assertTrue(control.resume())
            self.assertTrue(control.run_if_active(macro.renew))
            control.request_stop()
            self.assertEqual(messages[-1][1], 0x101)
        finally:
            control.set_release_input(None)
            macro.close()

    def test_ground_grace_is_bounded_and_explicit_block_releases(self):
        class Macro:
            def __init__(self): self.events = []
            def renew(self): self.events.append('renew'); return True
            def release(self): self.events.append('release')
        clock = [10.]
        macro = Macro()
        gate = GroundMacroGate(macro, RunControl(), clock=lambda: clock[0])
        self.assertFalse(gate.update(allowed=True, ground_frames=1, captured_at=10.))
        self.assertTrue(gate.update(allowed=True, ground_frames=2, captured_at=10.))
        clock[0] = 10.10
        self.assertTrue(gate.update(allowed=True, ground_frames=0, captured_at=10.10))
        clock[0] = 10.151
        self.assertFalse(gate.update(allowed=True, ground_frames=0, captured_at=10.151))
        self.assertEqual(macro.events[-1], 'release')
        self.assertTrue(gate.update(allowed=True, ground_frames=2, captured_at=10.151))
        self.assertFalse(gate.update(allowed=False, ground_frames=2, captured_at=10.151))
        self.assertFalse(gate.update(allowed=True, ground_frames=2, captured_at=9.))

    def test_ground_gate_records_distinct_release_reasons(self):
        class Macro:
            def __init__(self): self.reasons = []
            def note_stop_reason(self, reason, captured_at): self.reasons.append((reason,captured_at))
            def renew(self): return True
            def release(self): pass
        clock = [10.]
        macro = Macro()
        gate = GroundMacroGate(macro, RunControl(), clock=lambda:clock[0])
        gate.update(allowed=True,ground_frames=2,captured_at=10.)
        clock[0] = 10.151
        gate.update(allowed=True,ground_frames=0,captured_at=10.151)
        self.assertEqual(macro.reasons[-1][0],'gate_ground_expired')
        gate.update(allowed=False,ground_frames=2,captured_at=10.151)
        self.assertEqual(macro.reasons[-1][0],'gate_disallowed')
        gate.update(allowed=True,ground_frames=2,captured_at=9.)
        self.assertEqual(macro.reasons[-1][0],'gate_stale_frame')


if __name__ == '__main__':
    unittest.main()
