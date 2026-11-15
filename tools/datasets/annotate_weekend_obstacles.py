"""Export manually reviewed race boxes as COCO, YOLO and annotated previews."""
from ninja3_assistant.paths import project_root
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont


ROOT = project_root()
SOURCE = ROOT/'data/evidence/2026-11-15/weekend-passability'
OUTPUT = ROOT/'data/evidence/2026-11-15/weekend-obstacle-labels'
CLASSES = ('gate','plant','spike','bomb','pit')
NAMES = ('木栏','带刺植物','地刺','悬空炸弹','坑洞')
# Box convention: x, y, width, height in original 1280x720 frames.
# Last value marks a clipped/occluded object; no invisible extent is invented.
CASES = {
    'frame-6.png': [(0,699,436,142,114,False),(0,0,585,146,54,True)],
    'frame-6.1.png': [(0,615,446,142,111,True),(0,0,604,66,39,True)],
    'frame-18.png': [(1,947,390,143,176,False)],
    'frame-18.2.png': [(1,899,389,141,176,False)],
    'frame-20.5.png': [(1,653,389,137,177,True)],
    'frame-24.png': [(3,868,523,56,58,True),(3,1107,391,56,94,False),
                     (4,704,535,576,185,False)],
    'frame-30.png': [(3,1045,344,56,90,False)],
    'frame-32.png': [(2,758,464,42,96,False),(3,149,293,56,91,False),
                     (3,507,272,56,92,False),(3,984,286,57,93,False)],
    'frame-32.1.png': [(2,690,464,39,96,True),(2,1228,465,44,95,False),
                       (3,76,290,55,92,False),(3,435,272,56,91,False),
                       (3,910,287,57,93,False)],
    'flat-road-players.png': [],
}


def main():
    for name in ('images','labels','preview'):
        (OUTPUT/name).mkdir(parents=True,exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    colours = ('#ff4545','#f7a529','#ff65ce','#ffe849','#4ee5ff')
    coco = {'info':{'description':'Manually reviewed visible obstacle boxes. No automatic pseudo-labels. '
                    'Pit denotes a terrain gap region, not a sprite. Skill portals, players and item boxes are negatives.',
                   'source_video':'data/weekend-race/trial/20261114-143752/race.mp4',
                   'annotation_scope':'Visible world obstacles; UI depictions excluded. Occlusion flags retained.'},
            'images':[],'annotations':[],
            'categories':[{'id':i,'name':name,'display_name':NAMES[i]} for i,name in enumerate(CLASSES)]}
    previews = []
    for image_id,(name,boxes) in enumerate(CASES.items(),1):
        source = SOURCE/name if name != 'flat-road-players.png' else SOURCE.parent/'weekend-race'/name
        shutil.copy2(source,OUTPUT/'images'/name)
        image = Image.open(source).convert('RGB')
        width,height = image.size
        coco['images'].append({'id':image_id,'file_name':'images/'+name,'width':width,'height':height,
                               'source':source.relative_to(ROOT).as_posix()})
        draw = ImageDraw.Draw(image)
        labels = []
        for category,x,y,w,h,occluded in boxes:
            assert 0 <= x < x+w <= width and 0 <= y < y+h <= height
            coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':image_id,
                'category_id':category,'bbox':[x,y,w,h],'area':w*h,'iscrowd':0,'occluded':occluded})
            labels.append(f'{category} {(x+w/2)/width:.6f} {(y+h/2)/height:.6f} {w/width:.6f} {h/height:.6f}')
            colour = colours[category]
            draw.rectangle((x,y,x+w,y+h),outline=colour,width=3)
            label = NAMES[category]+('（遮挡/截断）' if occluded else '')
            ty = max(0,y-30)
            draw.rectangle((x,ty,min(width,x+draw.textlength(label,font=font)+8),ty+29),fill='#122033')
            draw.text((x+3,ty),label,font=font,fill=colour)
        if not boxes:
            draw.text((260,80),'负样本：玩家、特效和道具不是障碍',font=font,fill='#ffe849')
        (OUTPUT/'labels'/(Path(name).stem+'.txt')).write_text('\n'.join(labels)+'\n',encoding='utf-8')
        image.save(OUTPUT/'preview'/name)
        image.thumbnail((640,360))
        previews.append(image)
    sheet = Image.new('RGB',(1280,360*((len(previews)+1)//2)),'#122033')
    for i,image in enumerate(previews):
        sheet.paste(image,((i%2)*640,(i//2)*360))
    sheet.save(OUTPUT/'标注总览.jpg',quality=92)
    (OUTPUT/'annotations.coco.json').write_text(json.dumps(coco,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'classes.txt').write_text('\n'.join(CLASSES)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUTPUT),'images':len(CASES),'boxes':len(coco['annotations'])}))


if __name__ == '__main__':
    main()
