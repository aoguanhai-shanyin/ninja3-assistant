"""Page IDs and recognized controls for the reverse-S bounty menu."""
from ninja3_assistant.game_ui import PageAction, PageObservation
from ninja3_assistant.vision.game_update import game_update_banner


FILTER_POINTS = {'s_on': (876, 171), 'a_off': (766, 171), 'ss_off': (978, 171)}
ACTION_NAMES = {
    'home': 'open_hall',
    'hall': 'open_bounty',
    'menu': 'open_reverse',
    'invite': 'decline_invite',
    'confirm': 'confirm_settlement',
    's_on': 'enable_s_filter',
    'a_off': 'disable_a_filter',
    'ss_off': 'disable_ss_filter',
}


def join_control(screen, previous=None):
    if not screen.board or (screen.a, screen.s, screen.ss) != ('off', 'on', 'off'):
        return None
    if previous is None:
        return screen.join
    if screen.join and max(abs(a-b) for a, b in zip(screen.join, previous)) <= 8:
        return screen.join
    for left, top in screen.card_cells:
        if left <= previous[0] < left + 360 and top <= previous[1] < top + 176:
            if screen.join and left <= screen.join[0] < left + 360 and top <= screen.join[1] < top + 176:
                return screen.join
            return previous
    return None


def _menu_page(screen, previous_join):
    if screen.invite:
        return 'game.invite', {'invite': screen.invite}
    if screen.menu:
        return 'reverse_bounty.menu', {'menu': screen.menu}
    if screen.result:
        return 'reverse_bounty.result', {'confirm': screen.result}
    if screen.network_error:
        return 'game.network_retry', {'network_retry': screen.network_retry}
    if screen.login:
        return 'game.login', {'login': screen.login}
    if screen.home:
        return 'game.home', {'home': screen.home}
    if screen.hall:
        return 'game.hall', {'hall': screen.hall}
    if screen.board:
        return 'reverse_bounty.list', {
            's_on': FILTER_POINTS['s_on'] if screen.s == 'off' else None,
            'a_off': FILTER_POINTS['a_off'] if screen.a == 'on' else None,
            'ss_off': FILTER_POINTS['ss_off'] if screen.ss == 'on' else None,
            'refresh': screen.refresh,
            'join': join_control(screen, previous_join),
        }
    if screen.ready or screen.prepared:
        return 'reverse_bounty.room', {'ready': screen.ready}
    return None, {}


def reverse_observation(screen, frame, generation, captured_at, *, previous_join=None):
    page, points = _menu_page(screen, previous_join)

    def current(image):
        return getattr(image, 'shape', None) != (720, 1280, 3) or not game_update_banner(image)

    controls = {}
    for operation, point in points.items():
        if point is not None:
            name = ACTION_NAMES.get(operation, operation)
            hold = .015 if operation == 'join' else .07
            controls[name] = PageAction(name, point, guard=current, hold_seconds=hold)
    return PageObservation(page, frame, generation, captured_at, controls, detail=screen)
