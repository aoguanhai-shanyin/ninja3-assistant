import unittest
from flow import Flow
from vision import Screen

REFRESH, JOIN, READY = (1146, 171), (400, 306), (1110, 664)

def board(**kwargs):
    return Screen(board=True, a=kwargs.get('a', 'off'), s=kwargs.get('s', 'on'),
                  ss=kwargs.get('ss', 'off'), refresh=kwargs.get('refresh'),
                  join=kwargs.get('join'), failed=kwargs.get('failed', False),
                  card_cells=kwargs.get('card_cells', ()))

class FlowTest(unittest.TestCase):
    def test_confirmed_training_allows_bounded_unrecognized_intro_until_battle(self):
        flow = Flow()
        self.assertEqual(flow.step(Screen(training=True), 0).action, 'wait')
        self.assertEqual(flow.step(Screen(), 1).action, 'wait')
        self.assertEqual(flow.step(Screen(), 20).action, 'wait')
        self.assertEqual(flow.step(Screen(game_loading=True), 30).action, 'wait')
        self.assertEqual(flow.step(Screen(), 44).action, 'wait')
        self.assertEqual(flow.step(Screen(battle_active=True), 44.5).action, 'wait')
        self.assertEqual(flow.phase, 'battle')

    def test_training_transition_timeout_is_absolute_even_when_loading_flickers(self):
        flow = Flow()
        flow.step(Screen(training=True), 0)
        flow.step(Screen(), 1)
        self.assertEqual(flow.step(Screen(game_loading=True), 20).action, 'wait')
        self.assertEqual(flow.step(Screen(), 30).action, 'wait')
        self.assertEqual(flow.step(Screen(game_loading=True), 44.9).action, 'wait')
        self.assertEqual(flow.step(Screen(game_loading=True), 45).action, 'reopen')
        self.assertEqual(flow.step(Screen(), 60.1).action, 'stop')

    def test_long_gap_after_training_does_not_grant_a_fresh_transition_window(self):
        flow = Flow()
        flow.step(Screen(training=True), 0)
        self.assertEqual(flow.step(Screen(), 50).action, 'reopen')

    def test_visible_training_page_refreshes_but_loading_does_not_extend_window(self):
        flow = Flow()
        flow.step(Screen(training=True),0)
        flow.step(Screen(training=True),100)
        self.assertEqual(flow.step(Screen(),101).action,'wait')
        self.assertEqual(flow.step(Screen(game_loading=True),144.9).action,'wait')
        self.assertEqual(flow.step(Screen(game_loading=True),145).action,'reopen')

    def test_known_pages_cancel_training_transition_and_keep_original_priority(self):
        screens = [(Screen(login=(640,510)),'login'), (Screen(home=(800,600)),'home'),
                   (Screen(hall=(1080,150)),'hall'), (Screen(menu=(1050,470)),'menu'),
                   (Screen(invite=(1180,160)),'invite'), (Screen(result=(640,670)),'stop'),
                   (Screen(network_error=True,network_retry=(640,540)),'network_retry'),
                   (Screen(login=(640,510),game_loading=True),'login'),
                   (Screen(ready=READY),'stop'), (Screen(prepared=True),'stop'),
                   (board(refresh=REFRESH),'refresh')]
        for screen, expected in screens:
            with self.subTest(screen=screen):
                flow = Flow()
                flow.step(Screen(training=True),0)
                flow.step(Screen(),1)
                self.assertEqual(flow.step(screen,2).action,expected)
                self.assertFalse(flow.training_armed)

    def test_unknown_without_confirmed_training_keeps_original_timeout(self):
        flow = Flow()
        self.assertEqual(flow.step(Screen(),0).action,'wait')
        self.assertEqual(flow.step(Screen(),15).action,'reopen')

    def test_login_and_home_navigation_continue_to_s_bounty(self):
        flow = Flow()
        flow.entering = True
        self.assertEqual(flow.step(Screen(), 0).action, 'wait')
        login = flow.step(Screen(login=(640, 514)), 1)
        self.assertEqual(login.action, 'login'); flow.completed(login, 1)
        self.assertEqual(flow.step(Screen(login=(640, 514)), 3).action, 'wait')
        self.assertEqual(flow.step(Screen(), 30).action, 'wait')
        home = flow.step(Screen(home=(814, 615)), 31)
        self.assertEqual(home.action, 'home'); flow.completed(home, 31)
        hall = flow.step(Screen(hall=(1045, 161)), 32)
        self.assertEqual(hall.action, 'hall'); flow.completed(hall, 32)
        menu = flow.step(Screen(menu=(1040, 470)), 33)
        self.assertEqual(menu.action, 'menu'); flow.completed(menu, 33)
        self.assertEqual(flow.step(board(refresh=REFRESH), 34).action, 'refresh')

    def test_network_retry_and_unknown_login_dialog_do_not_blindly_click(self):
        flow = Flow()
        network = Screen(network_error=True, network_retry=(638, 538))
        action = flow.step(network, 0)
        self.assertEqual(action.action, 'network_retry'); flow.completed(action, 0)
        self.assertEqual(flow.step(network, 2).action, 'wait')
        self.assertEqual(flow.step(network, 11).action, 'network_retry')
        self.assertEqual(flow.step(Screen(network_error=True), 12).action, 'wait')
        self.assertEqual(flow.step(Screen(), 120).action, 'wait')

    def test_only_join_s_after_confirmed_filters(self):
        flow = Flow()
        action = flow.step(board(a='on', s='off', ss='on', refresh=REFRESH, join=JOIN), 0)
        self.assertEqual(action.action, 's_on')
        flow.completed(action, .1)
        self.assertEqual(flow.step(board(a='on', s='off', ss='on'), .2).action, 'wait')
        action = flow.step(board(a='on', s='on', ss='on'), .3)
        self.assertEqual(action.action, 'a_off')
        flow.completed(action, .4)
        action = flow.step(board(a='off', s='on', ss='on'), .5)
        self.assertEqual(action.action, 'ss_off')
        flow.completed(action, .6)
        action = flow.step(board(refresh=REFRESH, join=JOIN), .7)
        self.assertEqual(action.action, 'join')

    def test_refresh_then_recognize_immediately_and_react_to_cooldown(self):
        flow = Flow()
        action = flow.step(board(refresh=REFRESH), 0); flow.completed(action, .1)
        self.assertEqual(flow.step(board(join=JOIN), .2).action, 'join')
        flow = Flow()
        action = flow.step(board(refresh=REFRESH), 0); flow.completed(action, .1)
        self.assertEqual(flow.step(board(refresh=REFRESH), .2).action, 'wait')
        self.assertEqual(flow.step(board(), .3).action, 'wait')
        self.assertEqual(flow.step(board(refresh=REFRESH), .4).action, 'refresh')

    def test_closed_team_just_after_refresh_waits_for_button_to_recover(self):
        flow = Flow()
        action = flow.step(board(refresh=REFRESH), 0); flow.completed(action, .1)
        action = flow.step(board(join=JOIN, refresh=REFRESH), .2); flow.completed(action, .3)
        self.assertEqual(flow.step(board(failed=True, refresh=REFRESH), .4).action, 'wait')
        self.assertEqual(flow.step(board(), .5).action, 'wait')
        self.assertEqual(flow.step(board(refresh=REFRESH), .6).action, 'refresh')

    def test_repeat_same_team_until_battle_then_refresh(self):
        flow = Flow()
        action = flow.step(board(refresh=REFRESH), 0); flow.completed(action, .1)
        action = flow.step(board(join=JOIN), .3); flow.completed(action, .4)
        for now in (.6, .9, 1.2):
            action = flow.step(board(join=JOIN), now)
            self.assertEqual(action.action, 'join')
            flow.completed(action, now)
        self.assertEqual(flow.step(board(join=(770, 306)), 1.5).action, 'wait')
        action = flow.step(board(join=JOIN, failed=True, refresh=REFRESH), 1.7)
        self.assertEqual(action.action, 'refresh')

    def test_keeps_clicking_when_card_moves_or_plus_is_obscured(self):
        flow = Flow()
        first = flow.step(board(join=(389, 306)), 0)
        flow.completed(first, .1)
        moved = flow.step(board(join=(400, 306), card_cells=((95, 198),)), .2)
        self.assertEqual((moved.action, moved.point), ('join', (400, 306)))
        flow.completed(moved, .3)
        covered = flow.step(board(card_cells=((95, 198),)), .4)
        self.assertEqual((covered.action, covered.point), ('join', (400, 306)))
        other = flow.step(board(join=(770, 306), card_cells=((95, 198), (465, 198))), .45)
        self.assertEqual((other.action, other.point), ('join', (400, 306)))
        self.assertEqual(flow.step(board(s='?', card_cells=((95, 198),)), .5).action, 'wait')
        self.assertEqual(flow.step(board(ss='on', card_cells=((95, 198),)), .6).action, 'wait')

    def test_do_not_ready_unknown_room_but_ready_after_join(self):
        self.assertEqual(Flow().step(Screen(ready=READY), 0).action, 'stop')
        self.assertEqual(Flow().step(Screen(prepared=True), 0).action, 'stop')
        flow = Flow()
        action = flow.step(board(refresh=REFRESH), 0); flow.completed(action, .1)
        action = flow.step(board(join=JOIN), .3); flow.completed(action, .4)
        action = flow.step(Screen(ready=READY), .7)
        self.assertEqual(action.action, 'ready')
        flow.completed(action, .8)
        self.assertEqual(flow.step(Screen(ready=READY), 1).action, 'wait')
        self.assertEqual(flow.step(Screen(prepared=True), 1.2).action, 'wait')
        self.assertEqual(flow.phase, 'await_battle')

    def test_battle_result_confirm_and_next_round(self):
        flow = Flow()
        action = flow.step(board(refresh=REFRESH), 0); flow.completed(action, .1)
        action = flow.step(board(join=JOIN), .3); flow.completed(action, .4)
        action = flow.step(Screen(ready=READY), .7); flow.completed(action, .8)
        self.assertEqual(flow.step(Screen(battle_loading=True), 1).action, 'wait')
        self.assertEqual(flow.phase, 'battle')
        self.assertEqual(flow.step(Screen(battle_active=True), 2).action, 'wait')
        result = Screen(result=(650, 678))
        action = flow.step(result, 3)
        self.assertEqual(action.action, 'confirm'); flow.completed(action, 3.1)
        self.assertEqual(flow.step(result, 4).action, 'wait')
        self.assertEqual(flow.successful_kills, 0)
        self.assertEqual(flow.step(board(refresh=REFRESH), 5).action, 'refresh')
        self.assertEqual(flow.successful_kills, 1)
        flow.step(board(refresh=REFRESH), 5.1)
        self.assertEqual(flow.successful_kills, 1)

    def test_counter_waits_for_return_and_does_not_double_count(self):
        flow = Flow()
        result = Screen(result=(650, 678), reverse_result=True)
        action = flow.step(result, 0); flow.completed(action, .1)
        self.assertEqual(flow.step(Screen(), 1).action, 'wait')
        self.assertEqual(flow.successful_kills, 0)
        invite = Screen(invite=(1182, 168))
        action = flow.step(invite, 2)
        self.assertEqual(action.action, 'invite'); flow.completed(action, 2.1)
        self.assertEqual(flow.step(invite, 2.2).action, 'wait')
        self.assertEqual(flow.step(result, 3).action, 'wait')
        self.assertEqual(flow.successful_kills, 0)
        flow.step(board(refresh=REFRESH), 4)
        self.assertEqual(flow.successful_kills, 1)

        unconfirmed = Flow()
        action = unconfirmed.step(result, 0); unconfirmed.completed(action, .1)
        self.assertEqual(unconfirmed.step(result, 9).action, 'stop')
        self.assertEqual(unconfirmed.successful_kills, 0)

    def test_recover_from_identified_reverse_result_and_ignore_other_result(self):
        flow = Flow()
        self.assertEqual(flow.step(Screen(result=(650, 678)), 0).action, 'stop')
        action = flow.step(Screen(result=(650, 678), reverse_result=True), 1)
        self.assertEqual(action.action, 'confirm')

    def test_invite_and_unknown_page_do_not_trigger_blind_taps(self):
        flow = Flow()
        action = flow.step(Screen(invite=(1182, 168)), 0)
        self.assertEqual(action.action, 'invite'); flow.completed(action, .1)
        self.assertEqual(flow.step(Screen(invite=(1182, 168)), .3).action, 'wait')
        self.assertEqual(flow.step(Screen(), 1).action, 'wait')
        self.assertEqual(flow.step(Screen(), 16.1).action, 'reopen')
        self.assertEqual(flow.step(Screen(), 31.2).action, 'stop')

    def test_filter_never_toggles_twice_without_confirmation(self):
        flow = Flow()
        action = flow.step(board(ss='on'), 0); flow.completed(action, .1)
        self.assertEqual(flow.step(board(ss='on'), 1).action, 'wait')
        self.assertEqual(flow.step(board(ss='on'), 4.2).action, 'stop')

if __name__ == '__main__':
    unittest.main()
