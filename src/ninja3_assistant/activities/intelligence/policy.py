"""Small, deterministic purchase heuristic for recognized intelligence cards.

Board positions are row-major. None means a confirmed empty slot, never an
unrecognized occupied slot. Attack values are catalogue base values. This is a
single-purchase estimate, not a combat simulator: unknown effects and effects
requiring placement history are left out.
"""
from dataclasses import dataclass
import re
from typing import Sequence


@dataclass(frozen=True)
class Card:
    id: str
    attack: int
    cost: int
    element: str = ''
    profession: str = ''
    effect: str = ''


ELEMENTS = {'火': 'fire', '水': 'water', '风': 'wind', '雷': 'thunder'}


def _text(card):
    return re.sub(r'[\s\[\]【】,，。；;]', '', card.effect)


def _attacks(board, round_number):
    attacks = [float(card.attack) if card else 0. for card in board]
    weapons = sum(card is not None and card.profession == 'weapon' for card in board)
    fire = sum(card is not None and card.element == 'fire' for card in board)
    thunder = sum(card is not None and card.element == 'thunder' for card in board)
    for index, card in enumerate(board):
        if card is None:
            continue
        text = _text(card)
        if '每上场1张武器卡牌攻击+2' in text:
            attacks[index] += 2 * weapons
        if '每上场1张火系卡自身攻击+2' in text:
            attacks[index] += 2 * fire
        if '若同行有3张水系卡自身攻击×2' in text and all(
                item and item.element == 'water' for item in board[index//3*3:index//3*3+3]):
            attacks[index] *= 2
        if '上场5张雷系卡后自身攻击×3' in text and thunder >= 5:
            attacks[index] *= 3
        if '第4/5回合自身攻击×3' in text and round_number >= 4:
            attacks[index] *= 3
    for index, card in enumerate(board):
        if card is None:
            continue
        text = _text(card)
        match = re.fullmatch(r'在场时同行其他([火水风雷])系卡攻击\+(\d+)', text)
        if match:
            for target, other in enumerate(board):
                if other and target != index and target//3 == index//3 and other.element == ELEMENTS[match[1]]:
                    attacks[target] += int(match[2])
        if '上场5张雷系卡后同行其他卡攻击+4' in text and thunder >= 5:
            for target, other in enumerate(board):
                if other and target != index and target//3 == index//3:
                    attacks[target] += 4
        match = re.fullmatch(r'每上场1种(属性|职业)卡同行其他卡攻击\+(\d+)', text)
        if match:
            field = 'element' if match[1] == '属性' else 'profession'
            groups = {getattr(other, field) for other in board if other}
            groups -= {'', 'none'}
            for target, other in enumerate(board):
                if other and target != index and target//3 == index//3:
                    attacks[target] += len(groups) * int(match[2])
    return attacks


def _placement_bonus(board, index, attacks):
    card, bonus = board[index], 0.
    text = _text(card)
    match = re.search(r'上场回合造成(\d+)点伤害', text)
    if match:
        bonus += int(match[1])
    if '上场回合造成自身攻击×3的伤害' in text:
        bonus += attacks[index] * 3
    match = re.search(r'上场回合造成([本][行列])卡攻击和(×2)?的伤害', text)
    if match:
        values = [power for target, power in enumerate(attacks)
                  if (target//3 == index//3 if match[1] == '本行' else target%3 == index%3)]
        bonus += sum(values) * (2 if match[2] else 1)
    match = re.search(r'上场回合造成([火水风雷])系卡×(\d+)的伤害', text)
    if match:
        bonus += sum(other is not None and other.element == ELEMENTS[match[1]] for other in board) * int(match[2])
    match = re.search(r'上场回合造成武器卡×(\d+)的伤害', text)
    if match:
        bonus += sum(other is not None and other.profession == 'weapon' for other in board) * int(match[1])
    return bonus


def _temporary_bonus(board, index):
    text = _text(board[index])
    match = re.fullmatch(r'同([行列])其他(?:([火水风雷])系)?卡攻击\+(\d+)持续2回合', text)
    if match:
        return sum(int(match[3]) for target, other in enumerate(board)
                   if other and target != index
                   and (target//3 == index//3 if match[1] == '行' else target%3 == index%3)
                   and (not match[2] or other.element == ELEMENTS[match[2]]))
    match = re.fullmatch(r'([火水风雷])系卡攻击\+(\d+)持续2回合', text)
    if match:
        return sum(int(match[2]) for other in board if other and other.element == ELEMENTS[match[1]])
    return 0.


def _late_damage(board, attacks, turn):
    return sum(attacks[index] * 3 for index, card in enumerate(board)
               if card and turn >= 4 and '第4/5回合造成自身攻击×3的伤害' in _text(card))


def _coin_damage(board, coins):
    return sum(coins * 4 for card in board if card and
               '第5回合造成剩余金币数×4的伤害' in _text(card))


def choose_purchase(board: Sequence[Card | None], shop: Sequence[Card | None],
                    coins: int, round_number: int) -> tuple[int, int] | None:
    """Choose one affordable known card and a confirmed empty cell.

    Earlier rounds value repeated attack per coin; the last round values the
    immediate attack gain. Placement-only damage and two-round row/column buffs
    are included for the new card. Fifth-round coin damage uses the observed balance;
    earlier future balances and existing temporary buffs are excluded. Call again
    only after a fresh screenshot confirms the purchase.
    """
    if len(board) != 9 or len(shop) != 5 or not 1 <= round_number <= 5:
        raise ValueError('expected a 3x3 board, five shop slots, and round 1..5')
    if coins < 0 or any(card is not None and not isinstance(card, Card) for card in (*board, *shop)):
        raise ValueError('use recognized Card values and confirmed empty slots')
    if any(card and (not card.id or card.attack < 0 or card.cost < 0) for card in board):
        raise ValueError('an unknown occupied board slot cannot be treated as empty')
    empty = [index for index, card in enumerate(board) if card is None]
    best, best_score = None, (0., 0., 0.)
    for shop_index, card in enumerate(shop):
        if card is None or not card.id or card.attack < 0 or card.cost < 0 or card.cost > coins:
            continue
        for cell in empty:
            placed = list(board)
            placed[cell] = card
            future, immediate = 0., 0.
            temporary = _temporary_bonus(placed, cell)
            for turn in range(round_number, 6):
                attacks = _attacks(placed, turn)
                before = _attacks(board, turn)
                gain = sum(attacks) + _late_damage(placed, attacks, turn) - sum(before) - _late_damage(board, before, turn)
                if turn < round_number + 2:
                    gain += temporary
                if turn == round_number:
                    immediate = gain + _placement_bonus(placed, cell, attacks)
                    if turn == 5:
                        immediate += _coin_damage(placed, coins-card.cost) - _coin_damage(board, coins)
                else:
                    future += gain
            score = (immediate + future / max(card.cost, 1), immediate, -card.cost)
            if score > best_score:
                best, best_score = (shop_index, cell), score
    return best
