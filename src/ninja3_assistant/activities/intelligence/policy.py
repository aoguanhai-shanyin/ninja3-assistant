"""Bounded budget-and-placement search for recognized intelligence cards.

Board positions are row-major. None means a confirmed empty slot, never an
unrecognized occupied slot. Attack values are catalogue base values. This is a
bounded beam search, not a combat simulator: unknown effects and existing
temporary buffs requiring placement history are left out.
"""
from dataclasses import dataclass
from functools import lru_cache
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
    name: str = ''


ELEMENTS = {'火': 'fire', '水': 'water', '风': 'wind', '雷': 'thunder'}


@lru_cache(maxsize=512)
def _text(card):
    return re.sub(r'[\s\[\]【】,，。；;]', '', card.effect)


def _attacks(board, round_number):
    attacks = [float(card.attack) if card else 0. for card in board]
    weapons = sum(card is not None and card.profession == 'weapon' for card in board)
    fire = sum(card is not None and card.element == 'fire' for card in board)
    thunder = sum(card is not None and card.element == 'thunder' for card in board)
    tigers = sum(card is not None and '忍村5虎' in _text(card) for card in board)
    for index, card in enumerate(board):
        if card is None:
            continue
        text = _text(card)
        if '上场2虎后自身攻击+2' in text and tigers >= 2:
            attacks[index] += 2
        if '上场5虎后自身攻击×4' in text and tigers >= 5:
            attacks[index] *= 4
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
        if '上场4虎后角色卡攻击+1' in text and tigers >= 4:
            for target, other in enumerate(board):
                if other and other.profession == 'character':
                    attacks[target] += 1
        match = re.fullmatch(r'在场时相邻卡攻击\+(\d+)(?:([火水风雷])系卡额外\+(\d+))?', text)
        weapon_aura = '在场时相邻武器卡攻击+2' in text
        if match or weapon_aura:
            for target, other in enumerate(board):
                if not other or abs(target//3-index//3)+abs(target%3-index%3) != 1:
                    continue
                if weapon_aura:
                    attacks[target] += 2 if other.profession == 'weapon' else 0
                else:
                    attacks[target] += int(match[1])
                    if match[2] and other.element == ELEMENTS[match[2]]:
                        attacks[target] += int(match[3])
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


def _income(board):
    elements = {c.element for c in board if c and c.element not in ('', 'none')}
    return sum((1 if '每回合结束金币+1' in _text(c) else 0) +
               (len(elements) if '每回合结束每有1种属性卡金币+1' in _text(c) else 0)
               for c in board if c)


def _special_win(board):
    armor = {c.id for c in board if c and _text(c) == '鬼斩之铠'}
    trigger = any(c and '若上场3种鬼斩之铠直接胜利' in _text(c) for c in board)
    return trigger and len(armor) >= 3


def choose_purchase(board: Sequence[Card | None], shop: Sequence[Card | None],
                    coins: int, round_number: int, *, explanation=None) -> tuple[int, int] | None:
    """Plan several distinct shop purchases within budget; execute only the first.

    A deterministic beam keeps three placements per shop subset. The caller can
    continue the explained plan while fresh observations still agree with it.
    Scores estimate damage through round 5;
    they are neither win probabilities nor guarantees of globally optimal play.
    """
    if len(board) != 9 or len(shop) != 5 or not 1 <= round_number <= 5:
        raise ValueError('expected a 3x3 board, five shop slots, and round 1..5')
    if coins < 0 or any(card is not None and not isinstance(card, Card) for card in (*board, *shop)):
        raise ValueError('use recognized Card values and confirmed empty slots')
    if any(card and (not card.id or card.attack < 0 or card.cost < 0) for card in board):
        raise ValueError('an unknown occupied board slot cannot be treated as empty')
    board = tuple(board)
    eligible = tuple(i for i,c in enumerate(shop) if c and c.id and c.attack >= 0 and 0 <= c.cost <= coins)
    if _special_win(board):
        eligible = ()  # Preserve an already complete victory condition.
    remaining = 6-round_number
    future_turns = remaining-1
    # Value future purchases using this shop's observed base attack/cost. The
    # exact future shop is unknown, so log this conversion as an estimate.
    future_budget = coins+_income(board)*future_turns
    samples = tuple(c for c in shop if c and c.id and c.attack >= 0 and 0 <= c.cost <= future_budget)
    ratios = sorted(c.attack/max(c.cost,1) for c in samples)
    coin_value = ratios[len(ratios)//2] if ratios else 1.
    coin_value = max(.5,min(2.,coin_value))
    costs = sorted(c.cost for c in samples if c.cost > 0)
    expected_cost = costs[len(costs)//2] if costs else 2

    @lru_cache(maxsize=None)
    def economy(state, balance):
        income = _income(state)
        earned = income*future_turns
        free_cells = sum(c is None for c in state)
        capacity = free_cells*expected_cost
        # A remaining cell is more scarce when fewer cells remain than future
        # buying rounds. Do not price early, plentiful space as a guaranteed
        # future card; use only held coins and income actually supplied by cards.
        pressure = min(1.,max(0.,(future_turns+1-free_cells)/max(1,future_turns))) if future_turns else 0.
        available = min(earned+balance*pressure,capacity)
        invested = available*coin_value*(future_turns+1)/2
        reserved = (available-min(earned,capacity))*coin_value*(future_turns+1)/2
        # Split the same budget between investment and final cat damage; a coin
        # assigned to future purchases must not also remain for cat damage.
        investing = invested+_coin_damage(state,balance+earned-available)
        saving = _coin_damage(state,balance+earned)
        value = max(investing,saving)
        if saving > investing:
            reserved = 0.
        return value,income,earned,invested,reserved,free_cells,pressure

    before_economy = economy(board,coins)[0]

    @lru_cache(maxsize=None)
    def power(state, turn):
        attacks = _attacks(state, turn)
        return sum(attacks)+_late_damage(state, attacks, turn)

    def damage(state):
        early = max(0,4-round_number)
        return early*power(state,round_number)+(remaining-early)*power(state,max(4,round_number))

    baseline = damage(board)
    # state, moves, spent, placement damage. Empty plan is a valid save-coins choice.
    initial = (board,(),0,0.)
    def evaluate(node):
        state,moves,spent,burst = node
        temporary = sum(_temporary_bonus(state,cell) for _,cell in moves)*min(2,remaining)
        economic = economy(state,coins-spent)[0]-before_economy
        # The ring grants refreshes at this round's END, never spend them now.
        refreshes = sum(2 for _,cell in moves if '上场回合结束时刷新+2' in _text(state[cell]))
        free_cells = sum(c is None for c in state)
        refresh_value = (min(refreshes,free_cells)*.5*coin_value*future_turns
                         if future_turns and free_cells else 0.)
        return damage(state)-baseline+burst+temporary+economic+refresh_value

    frontier = [initial]
    best, best_key = initial,(int(_special_win(board)),0.,0,0.)
    evaluated = 0
    for _ in range(min(len(eligible),sum(c is None for c in board))):
        candidates = {}
        for state,moves,spent,burst in frontier:
            used = {i for i,_ in moves}
            for i in eligible:
                card = shop[i]
                if i in used or spent+card.cost > coins:
                    continue
                for cell,occupied in enumerate(state):
                    if occupied is not None:
                        continue
                    placed = list(state)
                    placed[cell] = card
                    placed = tuple(placed)
                    added = _placement_bonus(placed,cell,_attacks(placed,round_number))
                    node = (placed,moves+((i,cell),),spent+card.cost,burst+added)
                    value = evaluate(node)
                    evaluated += 1
                    # Same board and purchased shop slots have identical future
                    # options; preserve the ordering with the strongest entry effects.
                    identity = (placed,tuple(sorted(used|{i})))
                    previous = candidates.get(identity)
                    if previous is None or value > previous[0]:
                        candidates[identity] = (value,node)
                    key = (int(_special_win(placed)),value,-node[2],node[3])
                    if key > best_key:
                        best,best_key = node,key
        if not candidates:
            break
        ranked = sorted(candidates.values(),key=lambda item:(int(_special_win(item[1][0])),item[0],-item[1][2],item[1][3]),reverse=True)
        # Preserve each affordable shop subset, including low-damage combo
        # pieces, so a visible special-victory set cannot be pruned by damage.
        counts,frontier = {},[]
        for _,node in ranked:
            subset = tuple(sorted(i for i,_ in node[1]))
            if counts.get(subset,0) < 3:
                frontier.append(node)
                counts[subset] = counts.get(subset,0)+1
    state,moves,spent,burst = best
    if explanation is not None:
        single = 0.
        for i,cell in moves:
            alone = list(board); alone[cell] = shop[i]; alone = tuple(alone)
            single += evaluate((alone,((i,cell),),shop[i].cost,
                                _placement_bonus(alone,cell,_attacks(alone,round_number))))
        explanation.update(budget=coins,spent=spent,remaining_coins=coins-spent,
            round=round_number,estimated_gain=round(best_key[1],2),
            synergy_gain=round(best_key[1]-single,2),placement_damage=round(burst,2),
            special_victory=_special_win(state),
            armor_types=sorted({c.id for c in state if c and _text(c) == '鬼斩之铠'}),
            income_per_turn=economy(state,coins-spent)[1],
            future_income=economy(state,coins-spent)[2],
            economic_gain=round(economy(state,coins-spent)[0]-before_economy,2),
            held_coin_value=round(economy(state,coins-spent)[4],2),
            held_coin_value_before=round(economy(board,coins)[4],2),
            empty_cells=economy(state,coins-spent)[5],
            empty_cells_before=economy(board,coins)[5],
            granted_refreshes=sum(2 for _,cell in moves if '上场回合结束时刷新+2' in _text(state[cell])),
            economy_estimate={'attack_per_coin':coin_value,'expected_card_cost':expected_cost,
                              'refresh_value_per_future_turn':.5*coin_value,
                              'space_pressure':economy(state,coins-spent)[6]},
            plan=[{'shop':i,'cell':cell,'card':shop[i].id,'name':shop[i].name or shop[i].id,'cost':shop[i].cost,
                   'effect':shop[i].effect} for i,cell in moves],
            search='subset-beam3',evaluated=evaluated,
            limitations=['按卡库基础攻击估算','未追踪已有临时增益和历史上场次数',
                         '未来商店收益为估算，末回合结算收入不算可用金币',
                         '持币机会收益按可见价格与空位稀缺度估算，不假定未来具体卡牌','相邻按上下左右计算'])
    return moves[0] if moves else None
