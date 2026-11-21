"""Named controls on the daily bounty menus."""
from ninja3_assistant.activities.daily.flow import DailyDecision


HALL_SEARCH_LEFT = (430, 400)
HALL_SEARCH_RIGHT = (850, 400)


FIXED_CONTROLS = {
    'home': {'open_hall': ((865, 600), None)},
    'bounty_hub': {
        'open_tasks': ((155, 675), (20, 630, 280, 710)),
        'open_daily': ((360, 400), None),
        'open_order': ((930, 400), None),
    },
    'bounty_daily': {
        'open_tasks': ((330, 670), (260, 635, 410, 710)),
        'open_free_quota': ((380, 172), None),
        'open_grade': ((730, 670), None),
        'quick_join': ((1120, 670), None),
        'create_room': ((875, 670), None),
    },
    'bounty_tasks': {'close_tasks': ((1025, 155), (995, 125, 1060, 185))},
    'bounty_leave': {'confirm_leave': ((810, 525), (530, 340, 760, 395))},
    'bounty_order': {
        'enable_s_filter': ((876, 171), None),
        'disable_ss_filter': ((978, 171), None),
        'open_owned_orders': ((1115, 670), None),
    },
    'bounty_own_empty': {'close_owned_orders': ((1115, 670), None)},
    'bounty_recruit': {'close_recruitment': ((750, 350), (810, 25, 1270, 85))},
    'bounty_room': {
        'leave_room': ((65, 50), None),
        'leave_wrong_grade': ((65, 50), (280, 0, 360, 85)),
    },
}

OBSERVED_CONTROLS = {
    'decline_invite': ('point', (340, 475, 940, 585)),
    'confirm_settlement': ('confirm', None),
    'dismiss_reward': ('confirm', None),
    'claim_task': ('task_claim', (800, 235, 970, 425)),
    'open_bounty': ('entry', None),
    'confirm_grade': ('confirm', None),
    'close_free_quota': ('close', None),
    'join_daily': ('join', None),
    'join_order': ('join', None),
    'refresh': ('refresh', None),
    'confirm_create': ('create', None),
    'open_recruitment': ('invite_open', (520, 300, 755, 405)),
    'start_battle': ('start', None),
    'ready': ('ready', None),
    'login': ('login', None),
}


def menu_decision(screen, operation, status, *, value=None):
    if operation == 'find_entry':
        left, right = HALL_SEARCH_LEFT, HALL_SEARCH_RIGHT
        start, end = (right, left) if value == 'right' else (left, right)
        return DailyDecision('swipe', start, status, end, operation=operation, value=value)

    guard = free_kind = None
    if operation == 'accept_invite':
        point, guard = screen.invitation.accept, (120, 185, 940, 585)
    elif operation == 'choose_grade':
        point = screen.grade_choices.get(value)
    elif operation == 'claim_free_quota':
        point, guard, free_kind = screen.free_claim, screen.free_guard, 'bounty'
    elif operation == 'switch_recruitment_tab':
        point, guard = (885 if value == 'friend' else 1040, 60), (810, 25, 1270, 85)
    elif operation == 'invite_teammate':
        point = next((point for name, point in screen.recruit_candidates if name == value), None)
        if point:
            guard = (910, point[1] - 35, 1275, point[1] + 35)
    elif operation == 'back_to_hub':
        point = (65, 50)
    elif operation == 'back':
        point = screen.back
    elif operation in OBSERVED_CONTROLS:
        field, guard = OBSERVED_CONTROLS[operation]
        point = getattr(screen, field)
    else:
        point, guard = FIXED_CONTROLS[screen.page][operation]
    return DailyDecision('tap', point, status, free_kind=free_kind, guard=guard,
                         operation=operation, value=value)
