"""Daily chores use the shared task planner and native input executor."""

CHORES = {
    'monthly': '每月日签到（最多25天）',
    'limited': '限时签到',
    'blessing': '家族免费祈福',
    'weapon': '免费武器抽取',
    'curse': '免费咒印抽取',
    'bounty_shop': '悬赏商店忍币购买',
    'daily_shop': '每日商店忍币购买',
    'house': '小屋领取碎片并续练',
    'bounty': '每日悬赏清理',
    'bounty_s': 'S级悬赏清理（累计最多7次）',
    'friend_rice': '好友饭团领取与赠送',
    'village': '忍村试炼扫荡',
    'rank': '忍阶每日任务与每周道具',
    'pet_feed': '通灵兽每日喂养',
    'arena_reward': '竞技场奖励领取',
    'landlady_quiz': '老板娘答题',
}

PET_FEED_REWARDS = {'exp': '经验', 'coins': '忍币', 'hammers': '强化锤'}
DEFAULT_PET_FEED_REWARD = 'exp'
