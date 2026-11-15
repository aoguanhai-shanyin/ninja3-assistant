"""Extract the visually transcribed activity catalog from saved local screenshots.

Offline only: this module never imports the emulator or controller.
"""
from pathlib import Path
import hashlib
import json

import cv2


ROOT = Path(__file__).resolve().parents[1]
LIVE = Path('data/intelligence-simulation/live')

# Each row was reviewed against the complete card faces, not OCR output.
# Fields: name, base attack, price, element, effect, copies.
ROWS = [
    ('20261113-140356-cards-01.png', 156, 'character', [
        ('炎魂·小黑', 1, 1, 'fire', '忍村5虎。上场2虎后自身攻击+2', 1),
        ('龙焰·小黑', 1, 2, 'fire', '忍村5虎。上场4虎后[角色]卡攻击+1', 1),
        ('神乐·琳', 1, 1, 'water', '忍村5虎。上场2虎后自身攻击+2', 1),
        ('雪舞·琳', 1, 2, 'water', '忍村5虎。上场4虎后[角色]卡攻击+1', 1),
        ('鬼狩·苍牙', 1, 1, 'wind', '忍村5虎。上场2虎后自身攻击+2', 1),
        ('绯斩·苍牙', 1, 2, 'wind', '忍村5虎。上场4虎后[角色]卡攻击+1', 1),
    ]),
    ('20261113-140356-cards-01.png', 383, 'character', [
        ('雷兽·阿力', 1, 1, 'thunder', '忍村5虎。上场2虎后自身攻击+2', 1),
        ('超威·阿力', 1, 2, 'thunder', '忍村5虎。上场4虎后[角色]卡攻击+1', 1),
        ('神炎·小椒', 1, 1, 'fire', '忍村5虎。上场2虎后自身攻击+2', 1),
        ('星闪·小椒', 1, 2, 'thunder', '忍村5虎。上场4虎后[角色]卡攻击+1', 1),
        ('双焰·小椒', 1, 3, 'fire', '忍村5虎。上场5虎后自身攻击×4', 1),
        ('剑心·卫鲤', 2, 2, 'fire', '每上场1张[武器]卡牌，攻击+2', 2),
    ]),
    ('20261113-140430-cards-03.png', 108, 'character', [
        ('霜刃·伊鹤', 1, 3, 'water', '第4/5回合，造成自身攻击×3的伤害', 2),
        ('龙吟·洛青', 3, 2, 'water', '若同行有3张[水系]卡，自身攻击×2', 2),
        ('极刃·血影', 2, 3, 'wind', '第4/5回合，自身攻击×3', 2),
        ('夜叉·隼白', 6, 6, 'wind', '斩！上场回合造成15点伤害', 2),
        ('瞬翎·伊鹤', 2, 2, 'wind', '上场回合造成[武器]卡×4的伤害', 2),
        ('猫太·老板娘', 1, 2, 'wind', '第5回合，造成剩余金币数×4的伤害', 1),
    ]),
    ('20261113-140430-cards-03.png', 334, 'character', [
        ('雷羿·银枭', 2, 3, 'thunder', '上场5张[雷系]卡后，自身攻击×3', 2),
        ('鬼斩·紫原', 3, 2, 'thunder', '若上场3种[鬼斩之铠]，直接胜利', 1),
        ('樱火·荧', 1, 2, 'fire', '同列其他[火系]卡攻击+3，持续2回合', 1),
        ('寒绡·兮兰', 1, 2, 'water', '同列其他[水系]卡攻击+3，持续2回合', 1),
        ('乘霄·戌时', 1, 2, 'wind', '同列其他[风系]卡攻击+3，持续2回合', 1),
        ('月晖·小夜', 1, 2, 'thunder', '同列其他[雷系]卡攻击+3，持续2回合', 1),
    ]),
    ('20261113-140528-cards-04b.png', 218, 'character', [
        ('冰羽·兮兰', 1, 2, 'water', '同列其他卡攻击+2，持续2回合', 1),
        ('风铭·戌时', 1, 2, 'wind', '同行其他卡攻击+2，持续2回合', 1),
        ('奔雷·银枭', 1, 2, 'thunder', '[雷系]卡攻击+1，持续2回合', 1),
    ]),
    ('20261113-140511-cards-04.png', 247, 'weapon', [
        ('焚天', 2, 2, 'fire', '上场回合造成本列卡攻击和的伤害', 2),
        ('百炼', 1, 3, 'fire', '上场回合造成[火系]卡×4的伤害', 2),
        ('巨阙·缚日', 1, 3, 'fire', '上场回合造成10点伤害', 2),
        ('怒鲤双生', 1, 2, 'fire', '上场回合造成7点伤害', 2),
        ('封渊', 1, 2, 'water', '上场回合造成自身攻击×3的伤害', 2),
        ('丛云之霜羽', 1, 2, 'water', '上场回合造成7点伤害', 2),
    ]),
    ('20261113-140511-cards-04.png', 474, 'weapon', [
        ('赤芒', 2, 2, 'wind', '上场回合造成本行卡攻击和的伤害', 2),
        ('破云', 1, 2, 'wind', '上场回合造成7点伤害', 2),
        ('破雾·贯羽', 1, 3, 'thunder', '上场回合造成[雷系]卡×4的伤害', 2),
    ]),
    ('20261113-140544-cards-05.png', 388, 'treasure', [
        ('修罗覆面', 1, 2, 'fire', '在场时相邻卡攻击+1，[火系]卡额外+1', 2),
        ('菩提覆面', 1, 2, 'water', '在场时相邻卡攻击+1，[水系]卡额外+1', 2),
        ('应龙覆面', 1, 2, 'wind', '在场时相邻卡攻击+1，[风系]卡额外+1', 2),
        ('金刚覆面', 1, 2, 'thunder', '在场时相邻卡攻击+1，[雷系]卡额外+1', 2),
        ('风林火山战旗', 1, 3, 'none', '每上场1种属性卡，同行其他卡攻击+1', 2),
        ('超星太鼓', 1, 3, 'none', '每上场1种职业卡，同行其他卡攻击+1', 2),
    ]),
    ('20261113-140555-cards-06.png', 216, 'treasure', [
        ('轮回之戒', 2, 2, 'none', '上场回合结束时，刷新+2', 2),
        ('雷龙之鞘', 2, 2, 'thunder', '在场时相邻[武器]卡攻击+2', 2),
        ('VVVIP黑卡', 1, 1, 'wind', '每回合结束，金币+1', 2),
        ('惠比寿御守', 2, 3, 'wind', '每回合结束每有1种属性卡，金币+1', 2),
    ]),
    ('20261113-140610-cards-07.png', 175, 'beast', [
        ('黑龙', 1, 4, 'fire', '每上场1张[火系]卡，自身攻击+2', 1),
        ('云狐', 1, 4, 'water', '在场时同行其他[水系]卡攻击+4', 1),
        ('天鹰', 1, 4, 'wind', '上场回合造成本列卡攻击和×2的伤害', 1),
        ('雷龙', 1, 4, 'thunder', '上场5张[雷系]卡后同行其他卡攻击+4', 1),
    ]),
    ('20261113-140610-cards-07.png', 469, 'special', [
        ('鬼斩胸甲', 3, 2, 'thunder', '鬼斩之铠', 1),
        ('鬼斩臂甲', 2, 1, 'thunder', '鬼斩之铠', 2),
        ('鬼斩头盔', 2, 1, 'thunder', '鬼斩之铠', 2),
    ]),
]


def build():
    cards = []
    sources = {}
    target = ROOT / 'assets/intelligence-cards'
    target.mkdir(parents=True, exist_ok=True)
    for filename, y, profession, row in ROWS:
        source = LIVE / filename
        data = (ROOT / source).read_bytes()
        sources[source.as_posix()] = hashlib.sha256(data).hexdigest()
        frame = cv2.imread(str(ROOT / source))
        if frame is None or frame.shape[:2] != (720, 1280):
            raise ValueError(f'Unexpected source resolution: {source}')
        for column, (name, attack, cost, element, effect, copies) in enumerate(row):
            identifier = f'c{len(cards) + 1:03}'
            # Include the protruding attack badge and exclude the copy count.
            x = 153 + column * 166
            bbox = [x, y, x + 150, y + 181]
            template = Path('assets/intelligence-cards') / f'{identifier}.png'
            crop = frame[y:y + 181, x:x + 150]
            if crop.shape[:2] != (181, 150):
                raise ValueError(f'Incomplete crop: {name}')
            if not cv2.imwrite(str(ROOT / template), crop):
                raise OSError(f'Cannot save {template}')
            cards.append(dict(id=identifier, name=name, attack=attack, cost=cost,
                              element=element, profession=profession, effect=effect,
                              copies=copies, source=source.as_posix(), bbox=bbox,
                              template=template.as_posix()))
    if len(cards) != 53 or sum(c['copies'] for c in cards) != 81:
        raise ValueError('Catalog does not match the visible 81-card library')
    if len({c['name'] for c in cards}) != len(cards):
        raise ValueError('Duplicate card name')
    result = dict(schema_version=1, bbox_format='xyxy_exclusive',
                  resolution=[1280, 720], visually_transcribed=True,
                  total_unique=len(cards), total_copies=sum(c['copies'] for c in cards),
                  source_sha256=sources, cards=cards)
    output = ROOT / 'assets/intelligence-cards.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(catalog=str(output), cards=len(cards),
                          copies=result['total_copies']), ensure_ascii=False))


if __name__ == '__main__':
    build()
