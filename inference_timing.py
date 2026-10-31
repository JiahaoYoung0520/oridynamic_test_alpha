"""轻量墙钟计时；GPU阶段边界同步，不修改随机数或采样算子。"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


class TimingLog:
    def __init__(self, out, started=None):
        self.path = Path(out) / 'timing.json'
        self.started = time.perf_counter() if started is None else started
        self.sync = None
        self.data = dict(schema='oridynamic_inference_timing_v1', status='running', stages=[],
            device='cpu', cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            timing_method='perf_counter_wall_clock_cuda_sync_at_gpu_stage_boundaries')
        self._save()

    def _save(self):
        self.data['wall_seconds'] = time.perf_counter() - self.started
        temp = self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        os.replace(temp, self.path)

    def configure_cuda(self, synchronize, **device_info):
        self.sync = synchronize
        self.data.update(device_info)
        self._save()

    @contextmanager
    def stage(self, name, *, gpu=False, **details):
        if gpu and self.sync is None:
            raise RuntimeError('GPU计时未配置同步函数')
        if gpu:
            self.sync()
        started = time.perf_counter()
        label = ' '.join(f'{k}={v}' for k, v in details.items())
        print(f'[计时开始] {name} {label}', flush=True)
        status = 'complete'
        try:
            yield
            if gpu:
                self.sync()
        except BaseException:
            status = 'failed'
            raise
        finally:
            seconds = time.perf_counter() - started
            entry = dict(stage=name, seconds=seconds, status=status, cuda_synchronized=gpu, **details)
            self.data['stages'].append(entry)
            self._save()
            print(f'[计时] {name} {label} 耗时={seconds:.3f}s 状态={status}', flush=True)

    def finish(self, status, *, total_samples=0, **details):
        seconds = sum(e['seconds'] for e in self.data['stages']
                      if e['stage'] == 'ensemble_seed' and e['status'] == 'complete')
        self.data.update(status=status, total_samples=total_samples, ensemble_seconds=seconds,
            ensemble_seconds_per_sample=seconds / total_samples if total_samples else None,
            ensemble_samples_per_second=total_samples / seconds if seconds and total_samples else None,
            **details)
        self._save()
        self.data['wall_seconds_per_sample'] = self.data['wall_seconds'] / total_samples if total_samples else None
        # 此处固定同一个总墙钟值；不把摘要写盘自身计入耗时。
        temp = self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        os.replace(temp, self.path)
        print(f'[计时汇总] 状态={status} 总墙钟={self.data["wall_seconds"]:.3f}s '
              f'系综生成={seconds:.3f}s 样本数={total_samples}；详细记录={self.path}', flush=True)
        return self.data
