import unittest

import cv2

from ninja3_assistant.activities.xumi.teams import TeamSetup
from ninja3_assistant.paths import project_root


EVIDENCE = project_root()/'data/evidence/2026-11-19/xumi-152'
SCORED = project_root()/'data/runtime/xumi/runs/20261119-125304-988100'


@unittest.skipUnless((EVIDENCE/'021-own-selected.png').is_file(), 'local Xumi team replay unavailable')
class XumiTeamsTest(unittest.TestCase):
    def frame(self, name):
        return cv2.cvtColor(cv2.imread(str(EVIDENCE/name)), cv2.COLOR_BGR2RGB)

    def test_fixed_roster_waits_for_blue_acknowledgement_then_fills_second_slot(self):
        team = TeamSetup()
        self.assertEqual(team.step(self.frame('011-system.png'), 'fangcun_setup', 'fangcun').point, (561,564))
        self.assertEqual(team.step(self.frame('012-system-team.png'), 'team', 'fangcun').point, (399,288))
        roster = self.frame('013-system-roster.png')
        add = team.step(roster, 'roster', 'fangcun')
        self.assertEqual(add.point, (1045,214))
        team.completed(add)
        self.assertEqual(team.step(roster, 'roster', 'fangcun').action, 'wait')
        selected = self.frame('014-system-selected.png')
        close = team.step(selected, 'roster', 'fangcun')
        self.assertEqual(close.point, (624,76))
        team.completed(close)
        self.assertEqual(team.step(self.frame('015-system-team-filled.png'), 'team', 'fangcun').point, (632,288))
        self.assertEqual(team.step(selected, 'roster', 'fangcun').point, (1045,310))
        selected[193:236,998:1094] = roster[193:236,998:1094]
        self.assertEqual(team.step(selected, 'roster', 'fangcun').point, (1045,310))

    def test_owned_roster_waits_for_each_increase_and_saves_two_confirmed_members(self):
        team = TeamSetup()
        self.assertEqual(team.step(self.frame('018-own.png'), 'daqian_setup', 'daqian').point, (561,564))
        self.assertEqual(team.step(self.frame('019-own-team.png'), 'team', 'daqian').point, (399,288))
        roster = self.frame('020-own-roster.png')
        choose = team.step(roster, 'own_roster', 'daqian')
        self.assertEqual(choose.point, (765,377))
        team.completed(choose)
        self.assertEqual(team.step(roster, 'own_roster', 'daqian').action, 'wait')
        selected = self.frame('021-own-selected.png')
        second = team.step(selected, 'own_roster', 'daqian')
        self.assertEqual(second.point, (535,254))
        team.completed(second)
        self.assertEqual(team.step(selected, 'own_roster', 'daqian').action, 'wait')

        # The first filled bubble is real. Simulate a second receipt at its own position.
        with_two = selected.copy()
        with_two[261:318,237:294] = selected[261:318,108:165]
        self.assertEqual(team.step(with_two, 'own_roster', 'daqian').point, (720,597))

    @unittest.skipUnless((SCORED/'0060-fangcun_setup-tap.png').is_file(), 'local scored Xumi replay unavailable')
    def test_keeps_configured_leads_and_starts_at_first_difficulty(self):
        team = TeamSetup()
        setup = self.frame(SCORED/'0060-fangcun_setup-tap.png')
        ready = team.step(setup, 'fangcun_setup', 'fangcun')
        self.assertEqual((ready.action, ready.point), ('start', (1090,638)))
        one_member = self.frame(SCORED/'0061-team-tap.png')
        self.assertEqual(team.step(one_member, 'team', 'fangcun').action, 'wait')
        self.assertEqual(team.step(setup, 'fangcun_setup', 'daqian').action, 'wait')
        harder = setup.copy()
        harder[167:204,924:1014] = 0
        self.assertEqual(team.step(harder, 'fangcun_setup', 'fangcun').action, 'wait')

    def test_only_configures_the_team_with_an_empty_lead(self):
        team = TeamSetup()
        setup = self.frame('016-system-after.png')
        self.assertEqual(team.step(setup, 'fangcun_setup', 'fangcun').point, (562,647))
        configured = self.frame('015-system-team-filled.png')
        self.assertEqual(team.step(configured, 'team', 'fangcun').point, (120,277))


if __name__ == '__main__':
    unittest.main()
