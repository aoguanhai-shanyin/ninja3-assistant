"""A recognized invitation can transfer control at an activity checkpoint."""
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class InvitationHandoff:
    target: str


def clear_handled_invitation(control):
    """A confirmed non-invitation checkpoint ends duplicate suppression."""
    control.bounty_invite_suppressed = False


def bounty_handoff(frame, vision, ocr, control, *, safe, result=None):
    targets = getattr(control, 'bounty_invite_targets', {})
    if not safe or not targets or getattr(control, 'bounty_invite_suppressed', False):
        return None
    generation = control.generation()
    if result is None:
        result = ocr.read(frame, stop=control.stop)
    if result is None or control.stop.is_set() or control.paused() or generation != control.generation():
        return None
    invitation = vision.bounty_invitation(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), result)
    target = targets.get(invitation.grade) if invitation else None
    return InvitationHandoff(target) if target else None
