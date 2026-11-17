"""Home control combinations without private game templates or device input."""
import unittest
from unittest.mock import Mock

import numpy as np

from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.activities.daily.vision import DailyScreen, DailyVision


class HomeVisionTest(unittest.TestCase):
    def vision(self, anchors):
        vision=DailyVision.__new__(DailyVision)
        vision.invitations=Mock()
        vision.invitations.detect.return_value=None
        vision.match=Mock(side_effect=lambda frame,name,*args,**kwargs:anchors.get(name))
        vision.templates={}
        return vision

    def setUp(self):
        self.frame=np.zeros((720,1280,3),np.uint8)

    def test_home_requires_hall_and_summon_controls_together(self):
        for anchors in ({'home-hall':(814,614)},{'home':(582,676)},
                        {'home-summon':(677,687)},{}):
            with self.subTest(anchors=anchors):
                self.assertIsNone(self.vision(anchors).home(self.frame))

    def test_login_requires_both_button_and_server_controls(self):
        for anchors in ({},{'login-button':(640,515)},{'login-server':(751,432)}):
            with self.subTest(anchors=anchors):
                self.assertIsNone(self.vision(anchors).login(self.frame))
        vision=self.vision({'login-button':(640,515),'login-server':(751,432)})
        self.assertEqual(vision.navigation(self.frame),('login',(640,515)))
        self.assertEqual(vision.observe(self.frame,{'words':[]}).page,'login')

    def test_login_page_change_cancels_a_stale_click_in_either_direction(self):
        vision=self.vision({'login-button':(640,515),'login-server':(751,432)})
        home=DailyScreen(self.frame,{'words':[]},'home')
        self.assertFalse(vision.current_action(home,DailyDecision('tap',(677,687),'进入每日清理项目'),self.frame))
        login=DailyScreen(self.frame,{'words':[]},'login')
        decision=DailyDecision('tap',(640,515),'登录当前已选账号')
        self.assertTrue(vision.current_action(login,decision,self.frame))
        self.assertFalse(self.vision({}).current_action(login,decision,self.frame))

    def test_old_and_unlocked_bottom_layouts_provide_the_actual_summon_point(self):
        for name,point in (('home',(582,676)),('home-summon',(677,687))):
            with self.subTest(name=name):
                vision=self.vision({'home-hall':(814,614),name:point})
                screen=vision.observe(self.frame,{'words':[]})
                self.assertEqual(screen.page,'home')
                self.assertEqual(screen.home_summon,point)
                self.assertEqual(vision.navigation(self.frame),('home',None))

    def test_known_modal_is_handled_before_visible_home_background(self):
        for name,page in (('monthly-sidebar','monthly'),('reward-continue','reward_continue'),
                          ('reward','reward'),('anniversary','anniversary'),
                          ('ten-signin','ten-signin'),('rank-preview','rank-preview')):
            with self.subTest(name=name):
                vision=self.vision({'home-hall':(814,614),'home-summon':(677,687),name:(500,300)})
                self.assertEqual(vision.navigation(self.frame)[0],page)
        vision=self.vision({'home-hall':(814,614),'home-summon':(677,687),'anniversary':(500,300)})
        self.assertEqual(vision.observe(self.frame,{'words':[]}).page,'anniversary')

    def test_stale_back_is_rejected_for_both_home_layouts(self):
        screen=DailyScreen(self.frame,{'words':[]},'rank')
        decision=DailyDecision('tap',(75,50),'返回主页，继续每日悬赏')
        for name,point in (('home',(582,676)),('home-summon',(677,687))):
            with self.subTest(name=name):
                vision=self.vision({'home-hall':(814,614),name:point})
                self.assertFalse(vision.current_action(screen,decision,self.frame))

    def test_home_entry_points_follow_ocr_labels_that_move_between_accounts(self):
        vision=self.vision({'home-hall':(814,614),'home-summon':(677,687)})
        for point in ((230,228),(145,475)):
            with self.subTest(limited=point):
                x,y=point
                screen=vision.observe(self.frame,{'words':[
                    {'text':'答谢礼','bbox':[x-30,y-10,60,20]},
                    {'text':'忍卜屋','bbox':[110,215,70,22]},
                    {'text':'活动','bbox':[30,300,40,22]},
                    {'text':'商，店','bbox':[560,670,60,22]},
                ]})
                self.assertEqual(screen.home_points['limited'],point)
                self.assertEqual(screen.home_points['house'],(145,226))
                self.assertEqual(screen.home_points['monthly'],(50,311))
                self.assertEqual(screen.home_points['shop'],(590,681))

    def test_unreadable_limited_entry_has_no_fixed_coordinate_fallback(self):
        vision=self.vision({'home-hall':(814,614),'home-summon':(677,687)})
        screen=vision.observe(self.frame,{'words':[]})
        self.assertNotIn('limited',screen.home_points)
        self.assertEqual(screen.home_points['summon'],(677,687))
        self.assertEqual(screen.home_points['shop'],(573,687))

    def test_new_modal_cancels_a_click_planned_for_the_page_behind_it(self):
        screen=DailyScreen(self.frame,{'words':[]},'home')
        decision=DailyDecision('tap',(677,687),'进入每日清理项目')
        for name in ('summon-guide','shop-guide','reward','reward-continue'):
            with self.subTest(modal=name):
                self.assertFalse(self.vision({name:(500,300)}).current_action(screen,decision,self.frame))
        vision=self.vision({})
        vision.invitations.detect.return_value=(470,530)
        self.assertFalse(vision.current_action(screen,decision,self.frame))
        invitation=DailyScreen(self.frame,{'words':[]},'invite')
        self.assertTrue(vision.current_action(invitation,DailyDecision('tap',(470,530),'拒绝邀请'),self.frame))

    def test_next_page_requires_both_tutorial_borders(self):
        vision=self.vision({'guide-next':(1080,595)})
        self.assertIsNone(vision.guide_point(self.frame))
        self.frame[125]=(250,180,20)
        self.assertIsNone(vision.guide_point(self.frame))
        self.frame[594]=(250,180,20)
        self.assertEqual(vision.guide_point(self.frame),(1080,595))
        vision.match.side_effect=lambda *args,**kwargs:None
        s=vision.observe(self.frame,{'words':[{'text':'我知道了','bbox':[1020,580,120,30]}]})
        self.assertEqual(s.page,'paged-guide')
        self.assertEqual(s.point,(1080,595))


if __name__ == '__main__':
    unittest.main()
