"""Recognize the blocking client-update prompt from its frame and OCR."""


def game_update_banner(frame):
    if frame.shape != (720, 1280, 3):
        return False
    panel = frame[350,(100,1180)]
    if not ((panel.max(axis=1) < 65) & (panel[:,2] > panel[:,0])).all():
        return False
    return all((frame[rows, 25:1255:40] >= 230).all(axis=2).mean(axis=1).max() >= .9
               for rows in (slice(259,265), slice(452,457)))


def game_update_point(screen):
    if not game_update_banner(screen.frame):
        return None
    title = screen.hits(r'游[戏蜮]有更新',(485,282,790,349))
    text = screen.hits(r'重启客户端进行游[戏蜮]更新',(420,360,865,430))
    button = screen.hits('重启客户端',(510,484,775,552))
    return button[0].point if title and text and button else None
