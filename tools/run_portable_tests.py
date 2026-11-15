"""Public core regression suite: no game resources, models or emulator input."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

MODULES = (
    'test_activity_flow','test_activity_battle_flow','test_task_planner',
    'test_realtime_battle','test_experimental_input_scheduler',
    'test_mumu','test_mumu_macro','test_mumu_multitouch',
    'test_boss_policy','test_hayabusa_policy','test_hayabusa_async_pipeline',
    'test_hayabusa_recognition','test_hayabusa_fixed_opening','test_hayabusa_windup',
    'test_activity_ocr','test_native_touch','test_worker_activity',
)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.'+name for name in MODULES])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
