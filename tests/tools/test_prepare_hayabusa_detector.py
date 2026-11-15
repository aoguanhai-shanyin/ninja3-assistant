import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.datasets.prepare_hayabusa_detector import (CLASSES, CROP, apply_review,
    clipped_box, digest, prepare, sample_plan, select_reviews, write_labels)


class PrepareHayabusaDetectorTest(unittest.TestCase):
    def test_related_boards_and_all_replays_cannot_cross_splits(self):
        boards = [{'id':name,'families':['shuriken'],'clips':clips} for name,clips in (
            ('p1_shuriken_3',[{'start':0,'end':10,'title_intervals':[[0,2]],'replay_intervals':[[5,10]]}]),
            ('p1_shuriken_4',[{'start':10,'end':20,'replay_intervals':[]}]),
            ('p1_shuriken_34_variant',[{'start':20,'end':30,'replay_intervals':[]}]))]
        rows = sample_plan({'boards':boards},.5)
        self.assertEqual({row['split'] for row in rows},{'test'})
        self.assertEqual({row['group_id'] for row in rows},{'p1_shuriken_34'})
        self.assertTrue(all(2.5 < row['video_seconds'] < 4.5
            for row in rows if row['clip_id']=='p1_shuriken_3'))
        with self.assertRaises(ValueError):
            sample_plan({'boards':boards},float('nan'))

    def test_cropped_boxes_are_clipped_not_wrapped_and_empty_labels_are_retained(self):
        box = clipped_box('slash',[-20,80,50,520])
        self.assertEqual(box['xyxy'],[0,0,50,410])
        self.assertIsNone(clipped_box('fire',[100,501,140,540]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'negative.txt'
            write_labels(path,[])
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_text(),'')
            with self.assertRaises(ValueError):
                write_labels(path,[{'class':'fire','xyxy':[1,2,2000,300]}])

    def test_review_selection_covers_all_groups_and_keeps_empty_candidates(self):
        frames = [{'id':str(i),'group_id':str(i%6),'split':'val','boxes':[]}
                  for i in range(60)]
        rows = select_reviews(frames,'val',24)
        self.assertEqual(len(rows),24)
        self.assertEqual(len({row['group_id'] for row in rows}),6)
        self.assertTrue(all(not row['boxes'] for row in rows))

    def test_real_video_extraction_keeps_rule_negative_frames_without_claiming_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root/'sample.mp4'
            writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,36))
            self.assertTrue(writer.isOpened())
            for i in range(40):
                writer.write(np.full((36,64,3),i*5,np.uint8))
            writer.release()
            before = digest(video)
            catalog = root/'catalog.json'
            catalog.write_text(json.dumps({'boards':[{'id':'sample','families':['fire'],
                'clips':[{'start':0,'end':4}]}]}),encoding='utf-8')
            fake = SimpleNamespace(backend='test rules',reset=lambda:None,
                                   detect=lambda frame,now:(SimpleNamespace(active=False),[]))
            with patch('ninja3_assistant.combat.hayabusa.vision.HayabusaVision',return_value=fake):
                manifest = prepare(video,catalog,root/'out',interval=.75,cpu=True)
            self.assertEqual(manifest['status'],'completed')
            self.assertEqual(len(manifest['frames']),3)
            for row in manifest['frames']:
                self.assertFalse(row['reviewed'])
                self.assertEqual(row['boxes'],[])
                self.assertEqual((root/'out'/row['label_file']).read_text(),'')
                self.assertEqual(cv2.imread(str(root/'out'/row['file'])).shape,(410,1280,3))
            self.assertEqual(digest(video),before)
            with self.assertRaises(FileExistsError):
                prepare(video,catalog,root/'out')

    def test_only_explicitly_reviewed_holdouts_become_curated_ground_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = []
            for split in ('train','val','test'):
                relative = f'images/{split}/one.jpg'
                image = root/relative
                image.parent.mkdir(parents=True)
                image.write_bytes(split.encode())
                frames.append({'id':split,'split':split,'file':relative,
                    'label_file':f'labels/{split}/one.txt','image_sha256':digest(image),
                    'boxes':[],'weak_boxes':[],'reviewed':False})
            manifest = {'status':'completed','frames':frames,'classes':list(CLASSES),'crop_xyxy':list(CROP)}
            (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            review = {'classes':list(CLASSES),'crop_xyxy':list(CROP),
                      'frames':[{**row,'reviewed':row['split']=='val'} for row in frames[1:]]}
            review_file = root/'review.json'
            review_file.write_text(json.dumps(review),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Both val and test'):
                apply_review(root,review_file)
            self.assertFalse((root/'reviewed').exists())
            review['frames'][1]['reviewed'] = True
            review['frames'][1]['boxes'] = [{'class':'fire','xyxy':[10,20,70,60]}]
            review_file.write_text(json.dumps(review),encoding='utf-8')
            result = apply_review(root,review_file)
            self.assertEqual(result['counts'],{'train':1,'val':1,'test':1})
            self.assertFalse(result['frames'][0]['reviewed'])
            self.assertTrue(all(row['reviewed'] for row in result['frames'][1:]))
            self.assertEqual(result['frames'][2]['weak_boxes'],[])
            self.assertTrue((root/'reviewed/labels/test/one.txt').read_text().startswith('0 '))

    def test_training_corrections_and_exclusions_change_the_actual_training_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = []
            for ident, split in (('corrected','train'), ('excluded','train'),
                                 ('weak','train'), ('val','val'), ('test','test')):
                relative = f'images/{split}/{ident}.jpg'
                image = root/relative
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(ident.encode())
                frames.append({'id':ident,'split':split,'file':relative,
                    'label_file':f'labels/{split}/{ident}.txt','image_sha256':digest(image),
                    'expected_families':['slash'], 'reviewed':False,
                    'boxes':[{'class':'shuriken','xyxy':[10,20,70,60]}], 'weak_boxes':[]})
            (root/'manifest.json').write_text(json.dumps({'status':'completed','frames':frames}),encoding='utf-8')
            review = {'classes':list(CLASSES),'crop_xyxy':list(CROP),'frames':[
                {**frames[0],'reviewed':True,'boxes':[]},
                {**frames[1],'review_status':'excluded'},
                {**frames[3],'reviewed':True,'boxes':[]},
                {**frames[4],'reviewed':True,'boxes':[]}]}
            path = root/'review.json'
            path.write_text(json.dumps(review),encoding='utf-8')
            result = apply_review(root,path)
            self.assertEqual(result['counts'], {'train':2,'val':1,'test':1})
            self.assertEqual((root/'reviewed/labels/train/corrected.txt').read_text(), '')
            self.assertFalse((root/'reviewed/images/train/excluded.jpg').exists())
            self.assertTrue((root/'reviewed/labels/train/weak.txt').read_text().startswith('3 '))
            self.assertEqual(len({row['id'] for row in result['frames']}), 4)
            self.assertIn('train: "images/train"', (root/'reviewed/dataset.yaml').read_text())
            strong = apply_review(root, path, curated=root/'strong', reviewed_train_only=True)
            self.assertEqual(strong['counts'], {'train':1,'val':1,'test':1})
            self.assertTrue(all(row['reviewed'] for row in strong['frames']))
            self.assertFalse((root/'strong/images/train/weak.jpg').exists())


if __name__ == '__main__':
    unittest.main()
