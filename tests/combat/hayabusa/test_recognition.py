import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ninja3_assistant.combat.battle import BladeTrack
from ninja3_assistant.combat.hayabusa.recognition import (FEATURE_SIZE, FEATURE_VERSION, HayabusaBoardRecognizer,
    formation_features, mirror_features, sequence_costs)


def screen(**changes):
    values = dict(active=True, result=None, rope_y=430)
    values.update(changes)
    return SimpleNamespace(**values)


def projectile(kind='fire', x=700, y=360):
    return BladeTrack(1,x,y,60,28,-400,0,1,3,kind)


class BoardRecognitionTest(unittest.TestCase):
    def bank(self, duplicate=False, same_family=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)
        a = formation_features(screen(),[projectile()])
        b = a.copy() if duplicate else formation_features(screen(),
            [projectile('fire',x=300) if same_family else projectile('blackfire')])
        np.savez_compressed(path/'hayabusa-boards.npz',features=np.stack([a,b]),
            sequences=np.array([0,1]),labels=np.array([0,1]),progress=np.array([.3,.6]))
        data={'feature_version':FEATURE_VERSION,'min_confidence':.78,'min_margin':.055,
              'boards':[{'id':'fire_1','name':'鬼火一式','phase':'P1','phase_evidence':'context_inferred','families':['fire']},
                        {'id':'blackfire_1','name':'黑火一式','phase':'P2','phase_evidence':'context_inferred',
                         'families':['fire' if same_family else 'blackfire']}]}
        (path/'hayabusa-boards.json').write_text(json.dumps(data),encoding='utf-8')
        return HayabusaBoardRecognizer(path)

    def test_temporally_confirmed_formation_reports_name_and_inferred_phase(self):
        recognizer=self.bank()
        results=[recognizer.update(None,screen(),[projectile()],1+i*.1) for i in range(12)]
        result=results[-1]
        self.assertEqual(result.board_id,'fire_1')
        self.assertEqual(result.phase,'P1')
        self.assertEqual(result.name,'鬼火一式')
        self.assertEqual(result.phase_source,'context_inferred')
        self.assertAlmostEqual(result.progress,.3)
        self.assertIsNone(results[0].board_id)

    def test_identical_formations_do_not_force_a_board_or_phase(self):
        recognizer=self.bank(duplicate=True)
        for i in range(12):
            result=recognizer.update(None,screen(),[projectile()],1+i*.1)
            self.assertIsNone(result.board_id)
            self.assertIsNone(result.phase)

    def test_empty_tracks_inactive_and_result_screens_do_not_keep_stale_class(self):
        recognizer=self.bank()
        for i in range(12):
            recognizer.update(None,screen(),[projectile()],1+i*.1)
        for state,tracks in ((screen(),[]),(screen(active=False),[projectile()]),
                             (screen(result='victory'),[projectile()])):
            result=recognizer.update(None,state,tracks,3)
            self.assertIsNone(result.board_id)
            self.assertIsNone(result.phase)

    def test_board_features_cannot_read_hud_caption_background_or_clock_pixels(self):
        recognizer=self.bank()
        a=np.zeros((720,1280,3),np.uint8)
        b=np.random.default_rng(7).integers(0,256,a.shape,dtype=np.uint8)
        first=recognizer.update(a,screen(),[projectile()],1)
        recognizer.reset()
        second=recognizer.update(b,screen(),[projectile()],1)
        self.assertEqual(first,second)

    def test_mirror_preserves_geometry_and_is_an_involution(self):
        a=formation_features(screen(),[projectile(x=300)])
        b=formation_features(screen(),[projectile(x=980)])
        np.testing.assert_allclose(mirror_features(a),b,atol=1e-6)
        np.testing.assert_allclose(mirror_features(mirror_features(a)),a,atol=1e-6)

    def test_ordered_history_distinguishes_reversed_emission_sequences(self):
        tokens=np.eye(FEATURE_SIZE,dtype=np.float32)[:3]
        normal=sequence_costs(tokens,tokens,np.zeros(3,np.int32)).min()
        reverse=sequence_costs(tokens,tokens[::-1],np.zeros(3,np.int32)).min()
        self.assertLess(normal,.01)
        self.assertGreater(reverse,.4)
        slow=sequence_costs(tokens,np.repeat(tokens,2,axis=0),np.zeros(3,np.int32)).min()
        self.assertLess(slow,.02)

    def test_sequence_path_cannot_jump_between_clips(self):
        tokens=np.eye(FEATURE_SIZE,dtype=np.float32)[:3]
        independent=sequence_costs(tokens,tokens,np.arange(3,dtype=np.int32)).min()
        self.assertGreater(independent,.4)

    def test_confirmed_board_survives_short_gap_and_ends_after_visible_quiet(self):
        recognizer=self.bank()
        for i in range(12):
            result=recognizer.update(None,screen(),[projectile()],1+i*.1)
        self.assertTrue(result.locked)
        epoch=result.epoch
        short=recognizer.update(None,screen(),[],2.2)
        self.assertEqual(short.board_id,'fire_1')
        self.assertEqual(short.epoch,epoch)
        for now in (2.3,2.4,2.5,2.6,2.7,2.8):
            quiet=recognizer.update(None,screen(),[],now)
        self.assertIsNone(quiet.board_id)
        self.assertTrue(quiet.ended)

    def test_time_jump_clears_temporal_confirmation(self):
        recognizer=self.bank()
        for i in range(12): recognizer.update(None,screen(),[projectile()],1+i*.1)
        self.assertIsNone(recognizer.update(None,screen(),[projectile()],8).board_id)
        self.assertIsNone(recognizer.update(None,screen(),[projectile()],7).board_id)

    def test_missing_assets_are_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'缺少隼白板子参考库'):
                HayabusaBoardRecognizer(directory)

    def test_weak_same_family_candidate_does_not_restart_locked_mode(self):
        recognizer=self.bank(same_family=True)
        recognizer.locked=0
        recognizer.locked_confidence=.96
        recognizer.locked_margin=.2
        recognizer.last_supported=1.
        epoch=recognizer.epoch
        # This competitor exceeds initial acquisition thresholds but not switch hysteresis.
        with patch('ninja3_assistant.combat.hayabusa.recognition.sequence_costs',return_value=np.array([.26,.15],np.float32)):
            for i in range(18):
                result=recognizer.update(None,screen(),[projectile()],1+i*.1)
                self.assertEqual(result.board_id,'fire_1')
                self.assertEqual(result.epoch,epoch)
                self.assertFalse(result.onset)

    def test_strong_new_family_requires_consecutive_support_before_switch(self):
        recognizer=self.bank()
        recognizer.locked=0
        recognizer.locked_confidence=.96
        recognizer.locked_margin=.2
        recognizer.last_supported=1.
        with patch('ninja3_assistant.combat.hayabusa.recognition.sequence_costs',return_value=np.array([.6,.02],np.float32)):
            results=[recognizer.update(None,screen(),[projectile('blackfire')],1+i*.1) for i in range(7)]
        self.assertTrue(all(item.board_id=='fire_1' for item in results[:3]))
        self.assertEqual(results[-1].board_id,'blackfire_1')
        self.assertEqual(sum(item.onset for item in results),1)


if __name__=='__main__':
    unittest.main()
