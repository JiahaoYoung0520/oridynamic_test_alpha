"""CPU核验计时边界/失败状态/口径，不启动CUDA模型。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from inference_timing import TimingLog


class TimingTests(unittest.TestCase):
    def test_sync_order_and_summary(self):
        with tempfile.TemporaryDirectory() as root:
            timer = TimingLog(root)
            order = []
            timer.configure_cuda(lambda: order.append('sync'), device='test_mock_cuda')
            with timer.stage('ensemble_seed', gpu=True, seed=7, samples=2, chunk=1):
                order.append('work')
            with timer.stage('ensemble_seed', gpu=True, seed=8, samples=2, chunk=1):
                order.append('work')
            timer.finish('inference_complete', total_samples=4)
            got = json.loads((Path(root)/'timing.json').read_text())
            self.assertEqual(order, ['sync', 'work', 'sync'] * 2)
            expected = sum(e['seconds'] for e in got['stages'])
            self.assertEqual(got['ensemble_seconds'], expected)
            self.assertEqual(got['ensemble_seconds_per_sample'], expected/4)
            self.assertGreaterEqual(got['wall_seconds'], expected)

    def test_failure_is_not_completion(self):
        with tempfile.TemporaryDirectory() as root:
            timer = TimingLog(root)
            with self.assertRaisesRegex(ValueError, '原异常'):
                with timer.stage('native_features'):
                    raise ValueError('原异常')
            timer.finish('failed')
            got = json.loads((Path(root)/'timing.json').read_text())
            self.assertEqual(got['stages'][0]['status'], 'failed')
            self.assertEqual(got['status'], 'failed')
            self.assertIsNone(got['ensemble_seconds_per_sample'])

    def test_gpu_requires_explicit_sync(self):
        with tempfile.TemporaryDirectory() as root:
            timer = TimingLog(root)
            with self.assertRaises(RuntimeError):
                with timer.stage('gpu', gpu=True):
                    self.fail('未配置同步不应进入')

    def test_cpu_check_does_not_touch_cuda_or_rng(self):
        import torch
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as root, patch.object(torch.cuda, 'synchronize', side_effect=AssertionError):
            timer = TimingLog(root)
            with timer.stage('native_model_load'):
                pass
            timer.finish('cpu_check_complete')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
