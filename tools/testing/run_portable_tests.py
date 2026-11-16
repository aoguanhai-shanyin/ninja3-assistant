"""Public core regression suite: no game resources, models or emulator input."""
import sys
import unittest


MODULES = (
    'tests.activities.bounty.test_combat','tests.activities.bounty.test_daily_flow',
    'tests.activities.bounty.test_daily_runner',
    'tests.activities.daily.test_flow','tests.activities.daily.test_resources','tests.activities.daily.test_food_ocr',
    'tests.activities.intelligence.test_flow','tests.activities.intelligence.test_battle_flow',
    'tests.activities.intelligence.test_evidence','tests.activities.intelligence.test_purchase_planning',
    'tests.scheduling.test_task_planner','tests.scheduling.test_task_navigation',
    'tests.runtime.test_realtime_battle','tests.runtime.test_experimental_input_scheduler','tests.runtime.test_paths',
    'tests.device.test_mumu','tests.device.test_mumu_macro','tests.device.test_mumu_multitouch',
    'tests.combat.boss.test_policy','tests.combat.hayabusa.test_policy','tests.combat.hayabusa.test_async_pipeline',
    'tests.combat.hayabusa.test_recognition','tests.combat.hayabusa.test_fixed_opening','tests.combat.hayabusa.test_windup',
    'tests.activities.intelligence.test_ocr','tests.device.test_native_touch','tests.entrypoints.test_worker_activity',
)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
