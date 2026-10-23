"""Checks for rendering saved prediction JSON without running inference."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from anomnovic.visualize import visualize_predictions


class VisualizeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.output = self.root / 'output'
        self.predictions = self.root / 'predictions.json'
        self.ground_truth = self.root / 'ground_truth.json'
        for name in ('one.png', 'two.jpg'):
            image = np.full((150, 200, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(self.images / name), image))
        self.ground_truth.write_text(json.dumps({
            'images': [{'id': 7, 'file_name': 'one.png'}, {'id': 8, 'file_name': 'two.jpg'}],
            'categories': [{'id': 1, 'name': 'minibus'}],
            'annotations': [{'image_id': 7, 'category_id': 1, 'bbox': [10, 10, 12, 12]}],
        }), encoding='utf-8')

    def test_named_predictions_and_empty_images(self):
        self.predictions.write_text(json.dumps([
            {'image_name': 'one.png', 'category_name': 'cup', 'bbox': [30, 30, 40, 40], 'score': 0.8},
        ]), encoding='utf-8')
        count = visualize_predictions(str(self.images), str(self.predictions), str(self.output))
        self.assertEqual(count, 2)
        self.assertEqual(cv2.imread(str(self.output / 'one.png'))[30, 30].tolist(), [0, 0, 255])
        self.assertTrue((self.output / 'two.jpg').is_file())

    def test_id_predictions_overlay_and_explicit_class_names(self):
        self.predictions.write_text(json.dumps([
            {'image_id': 7, 'category_id': 1, 'bbox': [30, 30, 40, 40], 'score': 0.8},
        ]), encoding='utf-8')
        with patch('anomnovic.visualize.cv2.putText', wraps=cv2.putText) as put_text:
            count = visualize_predictions(
                str(self.images), str(self.predictions), str(self.output),
                ground_truth=str(self.ground_truth), show_ground_truth=True, classes=['minicar'])
        self.assertEqual(count, 2)
        self.assertIn('minicar: 0.80', [call.args[1] for call in put_text.call_args_list])
        image = cv2.imread(str(self.output / 'one.png'))
        self.assertEqual(image[10, 10].tolist(), [0, 0, 255])
        self.assertEqual(image[30, 30].tolist(), [255, 0, 0])
        self.assertFalse(np.array_equal(image, cv2.imread(str(self.images / 'one.png'))))

    def test_numeric_fallback_and_class_agnostic_name(self):
        self.predictions.write_text(json.dumps([
            {'image_name': 'one.png', 'category_id': 1, 'bbox': [30, 30, 40, 40], 'score': 0.8},
        ]), encoding='utf-8')
        with patch('anomnovic.visualize.cv2.putText', wraps=cv2.putText) as put_text:
            visualize_predictions(str(self.images), str(self.predictions), str(self.output))
        self.assertIn('class 1: 0.80', [call.args[1] for call in put_text.call_args_list])
        with patch('anomnovic.visualize.cv2.putText', wraps=cv2.putText) as put_text:
            visualize_predictions(str(self.images), str(self.predictions), str(self.output),
                                  classes=['object'], overwrite=True)
        self.assertIn('object: 0.80', [call.args[1] for call in put_text.call_args_list])

    def test_random_labels_and_sampling_repeat_with_seed(self):
        self.predictions.write_text(json.dumps([
            {'image_name': 'one.png', 'category_name': 'cup', 'bbox': [30, 30, 40, 40], 'score': 0.8},
        ]), encoding='utf-8')
        second_output = self.root / 'second_output'
        for directory in (self.output, second_output):
            count = visualize_predictions(str(self.images), str(self.predictions), str(directory),
                                          random_labels=True, seed=7, num_images=1)
            self.assertEqual(count, 1)
        first_files = list(self.output.iterdir())
        second_files = list(second_output.iterdir())
        self.assertEqual([path.name for path in first_files], [path.name for path in second_files])
        self.assertEqual(first_files[0].read_bytes(), second_files[0].read_bytes())

    def test_output_conflict_is_detected_before_writing(self):
        self.predictions.write_text('[]', encoding='utf-8')
        self.output.mkdir()
        (self.output / 'two.jpg').write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            visualize_predictions(str(self.images), str(self.predictions), str(self.output))
        self.assertFalse((self.output / 'one.png').exists())
        self.assertEqual((self.output / 'two.jpg').read_bytes(), b'keep')
        self.assertEqual(visualize_predictions(str(self.images), str(self.predictions),
                                               str(self.output), overwrite=True), 2)
        self.assertIsNotNone(cv2.imread(str(self.output / 'two.jpg')))

    def test_image_ids_require_ground_truth(self):
        self.predictions.write_text(json.dumps([
            {'image_id': 7, 'category_id': 1, 'bbox': [30, 30, 40, 40], 'score': 0.8},
        ]), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'without a ground-truth mapping'):
            visualize_predictions(str(self.images), str(self.predictions), str(self.output))
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
