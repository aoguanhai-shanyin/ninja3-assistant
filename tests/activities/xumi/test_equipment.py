from dataclasses import replace
from types import SimpleNamespace
import unittest

import cv2

from ninja3_assistant.activities.xumi.equipment import EquipmentSetup
from ninja3_assistant.activities.xumi.vision import XumiVision
from ninja3_assistant.paths import project_root


EVIDENCE = project_root()/'data/evidence/2026-11-19/xumi-152'
DUPLICATE = project_root()/'data/runtime/xumi/runs/20261119-123127-670400'
SUPPORT = project_root()/'data/runtime/xumi/runs/20261119-124706-859700'


@unittest.skipUnless((EVIDENCE/'033-own-weapon-assigned.png').is_file(), 'local Xumi equipment replay unavailable')
class XumiEquipmentTest(unittest.TestCase):
    def setUp(self):
        self.vision = XumiVision()
        self.gear = EquipmentSetup()

    def frame(self, name):
        return cv2.cvtColor(cv2.imread(str(EVIDENCE/name)), cv2.COLOR_BGR2RGB)

    def step(self, frame):
        return self.gear.step(frame, self.vision.scene(frame))

    def try_first_weapon(self):
        base = self.frame('029-own-equipment.png')
        choose = self.step(base)
        self.assertEqual(choose.point, (1010,215))
        self.gear.completed(choose)
        self.assertEqual(self.step(base).action, 'wait')

        detail = self.frame('031-own-weapon-selected.png')
        open_targets = self.step(detail)
        self.assertEqual(open_targets.point, (1134,573))
        self.gear.completed(open_targets)

        targets = self.frame('032-own-weapon-targets.png')
        scene = self.vision.scene(targets)
        other = replace(scene, equipment_targets=((378,345),(253,211),(253,495),(378,628)))
        self.assertEqual(self.gear.step(targets,other).action, 'wait')
        place = self.step(targets)
        self.assertEqual(place.point, (253,345))
        self.gear.completed(place)
        return base

    def test_first_placement_waits_for_the_exact_row_and_a_fresh_inventory(self):
        base = self.try_first_weapon()
        assigned = self.frame('033-own-weapon-assigned.png')
        wrong_row = base.copy()
        wrong_row[167:255,210:297] = assigned[301:389,210:297]
        self.assertEqual(self.step(wrong_row).action, 'wait')

        stale_inventory = assigned.copy()
        stale_inventory[177:466,972:1256] = base[177:466,972:1256]
        self.assertEqual(self.step(stale_inventory).action, 'wait')
        next_weapon = self.step(assigned)
        self.assertEqual((next_weapon.action,next_weapon.point), ('tap',(1010,215)))
        self.gear.completed(next_weapon)
        self.assertEqual(self.step(self.frame('031-own-weapon-selected.png')).action, 'wait')

    def test_saves_after_only_the_current_team_has_four_visible_weapons(self):
        self.try_first_weapon()
        complete = self.frame('033-own-weapon-assigned.png')
        # Fill only the rows above the separator; the other characters stay empty.
        for x in (253,378):
            for y in (211,345):
                complete[y-30:y+15,x-42:x+43] = complete[315:360,211:296]
        save = self.step(complete)
        self.assertEqual((save.action,save.point), ('gear_done',(1115,678)))
        self.gear.completed(save)
        self.assertEqual(self.step(complete).action, 'wait')
        untouched = EquipmentSetup().step(complete,self.vision.scene(complete))
        self.assertEqual((untouched.action,untouched.point), ('gear_done',(47,40)))

    @unittest.skipUnless((EVIDENCE/'053-own-team2-gear.png').is_file(), 'local team 2 equipment replay unavailable')
    def test_team_number_and_top_team_boundary_are_required(self):
        team_two = self.frame('053-own-team2-gear.png')
        scene = SimpleNamespace(**{**vars(self.vision.scene(team_two)), 'team_number': 2})
        self.assertEqual(EquipmentSetup(expected_team=1).step(team_two,scene).action, 'wait')
        gear = EquipmentSetup(expected_team=2)
        self.assertEqual(gear.step(team_two,scene).point, (1010,215))
        team_two[409:412,95:260] = team_two[405:408,95:260]
        self.assertEqual(gear.step(team_two,scene).action, 'wait')

    @unittest.skipUnless((SUPPORT/'0036-equipment-gear_done.png').is_file(), 'local marked-support replay unavailable')
    def test_remembers_the_marked_support_for_target_selection_but_rejects_unmarked_rows(self):
        gear = EquipmentSetup(expected_team=2)
        base = self.frame(SUPPORT/'0036-equipment-gear_done.png')
        base_scene = SimpleNamespace(**{**vars(self.vision.scene(base)), 'team_number': 2, 'support_rows': (495,)})
        choose = gear.step(base,base_scene)
        self.assertEqual((choose.action,choose.point), ('tap',(1010,215)))
        gear.completed(choose)

        detail = self.frame('031-own-weapon-selected.png')
        detail[175:250,973:1047] = base[175:250,973:1047]
        detail_scene = SimpleNamespace(**{**vars(self.vision.scene(detail)), 'team_number': 2})
        equip = gear.step(detail,detail_scene)
        self.assertEqual(equip.point, (1134,573))
        gear.completed(equip)

        target = self.frame('032-own-weapon-targets.png')
        target[175:250,973:1047] = base[175:250,973:1047]
        target_scene = SimpleNamespace(**{**vars(self.vision.scene(target)), 'team_number': 2})
        self.assertEqual(gear.step(target,target_scene).point, (253,495))
        target_scene.equipment_targets = ((253,628),(378,628))
        self.assertEqual(gear.step(target,target_scene).action, 'wait')

    @unittest.skipUnless((EVIDENCE/'045-own-secondary-assigned.png').is_file(), 'local secondary equipment replay unavailable')
    def test_primary_only_needs_a_glowing_secondary_and_accepts_single_star(self):
        primary_only = self.frame('042-own-gear-armed.png')
        choose = self.step(primary_only)
        self.assertEqual((choose.action,choose.point), ('tap',(1010,215)))
        self.gear.completed(choose)

        detail = self.frame('043-own-secondary-detail.png')
        equip = self.step(detail)
        self.assertEqual(equip.point, (1134,573))
        self.gear.completed(equip)

        targets = self.frame('044-own-secondary-target.png')
        place = self.step(targets)
        self.assertEqual(place.point, (378,345))
        self.gear.completed(place)
        assigned = self.frame('045-own-secondary-assigned.png')
        self.assertEqual(self.step(assigned).point, (1010,215))

        single_star = primary_only.copy()
        single_star[170:260,966:1059] = primary_only[376:466,966:1059]
        choose = EquipmentSetup().step(single_star, self.vision.scene(single_star))
        self.assertEqual((choose.action,choose.point), ('tap',(1010,215)))

    @unittest.skipUnless((DUPLICATE/'0017-equipment-wait.png').is_file(), 'local second weapon replay unavailable')
    def test_scaled_hammer_receipt_is_verified_without_rejecting_a_duplicate_top_icon(self):
        before = self.frame(DUPLICATE/'0014-equipment-tap.png')
        self.gear.completed(self.step(before))
        self.gear.completed(self.step(self.frame(DUPLICATE/'0015-equipment_detail-tap.png')))
        place = self.step(self.frame(DUPLICATE/'0016-equipment_target-tap.png'))
        self.assertEqual(place.point, (253,211))
        self.gear.completed(place)

        after = self.frame(DUPLICATE/'0017-equipment-wait.png')
        wrong_icon = after.copy()
        wrong_icon[167:255,210:297] = after[301:389,210:297]
        self.assertEqual(self.step(wrong_icon).action, 'wait')
        self.assertEqual(self.step(after).point, (1010,215))


if __name__ == '__main__':
    unittest.main()
