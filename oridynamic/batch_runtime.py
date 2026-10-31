"""两阶段常驻模型调度；复用发行版原生特征、anchor策略和完整采样算子。"""
from __future__ import annotations

from dataclasses import asdict
import copy
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

from oridynamic import inference_runtime as rt
from oridynamic import runtime_utils as runner
from oridynamic.inference_timing import TimingLog
from oridynamic.prepare import file_info, seed_list
from oridynamic.predict import select_runtime
from protenix.model.protenix import Protenix, update_input_feature_dict
from oridynamic.weights_io import load_weights
from oridynamic.complex_inputs import native_sample, proteins


def _cache_tree(value):
    """缓存仅保留张量和Python基本类型；不改变张量或源对象。"""
    # object数组的tolist()保留np.str_等标量；必须先于str/float判断转换。
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) in (str, int, float, bool, type(None), torch.Tensor):
        return value
    if type(value) is dict:
        return {_cache_tree(k): _cache_tree(v) for k, v in value.items()}
    if type(value) is list:
        return [_cache_tree(v) for v in value]
    if type(value) is tuple:
        return tuple(_cache_tree(v) for v in value)
    raise TypeError(f'不支持的缓存对象：{type(value).__name__}')


def pack_atoms(atoms):
    return _cache_tree(dict(coordinate=torch.from_numpy(atoms.coord.copy()),
        annotations={name:dict(dtype=atoms.get_annotation(name).dtype.str,
                              values=atoms.get_annotation(name).tolist())
                     for name in atoms.get_annotation_categories()},
        bonds=None if atoms.bonds is None else torch.tensor(atoms.bonds.as_array().astype(np.int64))))


def unpack_atoms(value):
    import biotite.structure as struc
    atoms = struc.AtomArray(len(value['coordinate']))
    atoms.coord = value['coordinate'].numpy().copy()
    for name, annotation in value['annotations'].items():
        atoms.set_annotation(name, np.asarray(annotation['values'], dtype=annotation['dtype']))
    if value['bonds'] is not None:
        atoms.bonds = struc.BondList(len(atoms), value['bonds'].numpy().astype(np.uint32))
    return atoms


def write_prepared(folder, features, atoms, identities, anchor, c0, contract):
    """只存CPU张量和基本类型；不是任意Python对象pickle，逐目标磁盘卸载O(N²)上下文。"""
    rt.require(runner.build_reference_identities(atoms) == identities, '缓存原子身份不符')
    payload = _cache_tree(dict(schema='oridynamic_prepared_target_v1', contract=contract,
        features=features, atoms=pack_atoms(atoms), identities=identities, anchor=anchor, c0=c0))
    rt.fingerprints(payload)  # 拒绝自定义对象/非张量数组；不在cache中存模型或训练样本。
    path = folder / 'prepared.pt'
    rt.require(not path.exists(), '拒绝覆盖prepared缓存')
    temp = path.with_suffix('.pt.tmp')
    torch.save(payload, temp)
    temp.replace(path)
    rt.save_json(folder / 'prepared.json', dict(contract=payload['contract'], payload=file_info(path)))


def read_prepared(folder, contract):
    record = json.loads((folder / 'prepared.json').read_text())
    rt.require(record['contract'] == contract, 'prepared输入/权重/协议不匹配')
    path = folder / 'prepared.pt'
    rt.require(file_info(path) == record['payload'], 'prepared文件已改变')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    rt.require(payload['schema'] == 'oridynamic_prepared_target_v1' and payload['contract'] == contract,
               'prepared payload协议不符')
    atoms = unpack_atoms(payload['atoms'])
    rt.require(runner.build_reference_identities(atoms) == payload['identities'], '缓存原子身份改变')
    return payload, atoms


class ResidentRuntime:
    def __init__(self, args, weights):
        self.a, self.weights = args, weights
        self.timer = args.timing
        self.results, self.contracts = [], {}
        self.native = self.model = self.saved = None
        self.load_counts = dict(native=0, codec=0, joint=0)
        self.device = torch.device(args.device)
        rt.require(self.device.type == 'cuda' and torch.cuda.is_available(), '批量推理需要CUDA；CPU请运行tests')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.configure_timer(self.timer)
        select_runtime(args.runtime_root)

    def configure_timer(self, timer):
        timer.configure_cuda(lambda: torch.cuda.synchronize(self.device), device=str(self.device),
                             gpu_name=torch.cuda.get_device_name(self.device))

    def load_base(self):
        rt.require(self.native is None and self.model is None, 'base只能加载一次')
        with self.timer.stage('joint_load_and_weight_hashes'):
            self.saved, self.hashes = rt.load_parent(self.weights)
            self.cfg = rt.make_config(self.a, self.saved)
            self.code = {name:file_info(self.a.runtime_root / name)['sha256'] for name in (
                'oridynamic/batch_predict.py', 'oridynamic/batch_runtime.py',
                'oridynamic/inference_runtime.py', 'oridynamic/codec_condition.py', 'oridynamic/local_msa.py', 'oridynamic/complex_inputs.py',
                'oridynamic/runtime_utils.py', 'protenix/model/modules/joint_ensemble.py')}
        with self.timer.stage('native_model_load'):
            runner.reset_template_probe_rng(101)
            self.native = Protenix(self.cfg)
            base = load_weights(self.weights['native'], 'native')
            self.native.load_state_dict({k.removeprefix('module.'):v for k,v in base.get('model', base).items()}, strict=True)
            del base
            self.native.eval()
            self.load_counts['native'] += 1
        with self.timer.stage('base_to_gpu', gpu=True):
            self.native.to(self.device)

    def target_args(self, target, phase):
        a = copy.copy(self.a)
        a.out = self.a.out / 'targets' / target['name']
        a.msa_a3m = target.get('msa_a3m')
        a.single_sequence = target['msa_mode'] == 'single_sequence'
        a.initial_structure = target.get('initial_structure')
        a.structure_chain = target.get('structure_chain')
        a.structure_chains = target.get('structure_chains')
        # 每目标两份独立阶段计时，避免anchor完成后等待其他目标的时间被计入本目标推理。
        timing_dir = a.out / phase
        timing_dir.mkdir(exist_ok=False)
        a.timing = TimingLog(timing_dir)
        self.configure_timer(a.timing)
        return a

    def prepare(self, target):
        rt.require(self.model is None, 'LoRA加载后禁止再生成base anchor/C0')
        a = self.target_args(target, 'anchor_phase')
        cfg = copy.deepcopy(self.cfg)
        cfg.input_json_path = str(a.out / 'input.json')
        cfg.dump_dir = str(a.out / 'native')
        rt.require(all(p.get('msa_a3m') for p in proteins(target)), '批量入口不允许隐式在线搜索')
        for identity in target.get('ligand_files', []):
            rt.require(file_info(Path(identity['path'])) == identity, '配体文件在输入检查后改变')
        with self.timer.stage('anchor_target', gpu=True, target=target['name']):
            features_cpu, atoms, identities = rt.prepare_features(a, cfg,
                [native_sample(target)])
            rt.trace_inputs(a, 'features', features_cpu)
            features = update_input_feature_dict(self.native.relative_position_encoding.generate_relp(rt.move(features_cpu, self.device)))
            with torch.no_grad(), rt.amp(a):
                with a.timing.stage('base_context', gpu=True, cycles=4):
                    runner.reset_template_probe_rng(101)
                    c0 = tuple(v.detach().cpu() for v in self.native.get_pairformer_output(
                        features, 4, inplace_safe=False, chunk_size=a.attention_chunk_size))
                if a.initial_structure:
                    with a.timing.stage('user_anchor_map'):
                        rt.require(file_info(Path(a.initial_structure)) == target['initial_structure_identity'],
                                   '用户初始结构在输入检查后改变')
                        anchor = rt.map_structure(a.initial_structure, atoms, a.structure_chain, a.structure_chains)
                    selection = dict(source='user_structure', input=target['initial_structure_identity'])
                else:
                    candidates, scores = [], []
                    for i in range(a.anchor_candidates):
                        runner.reset_template_probe_rng(a.anchor_seed+i)
                        with a.timing.stage('ptx_anchor', gpu=True, candidate=i+1, seed=a.anchor_seed+i):
                            pred = self.native(input_feature_dict=copy.deepcopy(features), label_full_dict=None,
                                label_dict=None, mode='inference', mc_dropout_apply_rate=0., disable_inplace=True)[0]
                            candidates.append(pred['coordinate'][0].detach().cpu().float())
                            scores.append(float(pred['summary_confidence'][0]['ranking_score']))
                            del pred
                    index = rt.choose_anchor(candidates, scores, len(atoms))
                    anchor = candidates[index]
                    selection = dict(source='ptx_predict', policy='native_ranking_score_top1', selected=index,
                        seeds=list(range(a.anchor_seed, a.anchor_seed+a.anchor_candidates)),
                        scores=[v if np.isfinite(v) else None for v in scores])
                    del candidates
            rt.trace_inputs(a, 'base_context', c0)
            rt.trace_inputs(a, 'anchor', anchor)
            contract = dict(target=target, weights=self.hashes, code=self.code, dtype=a.dtype,
                attention_backend=a.attention_backend, attention_chunk_size=a.attention_chunk_size,
                feature_seed=101, codec_cycles=4, anchor_cycles=10,
                native_sampling=cfg.sample_diffusion.to_dict(), anchor_selection=selection)
            with a.timing.stage('prepared_write'):
                write_prepared(a.out, features_cpu, atoms, identities, anchor, c0, contract)
                rt.write_coordinates(a.out / 'anchor.cif', atoms, identities, anchor[None])
                rt.save_json(a.out / 'resolved_config.json', cfg.to_dict())
            self.contracts[target['name']] = contract
            del features, features_cpu, c0, anchor
        a.timing.finish('anchor_and_base_context_complete')

    def load_joint(self):
        with self.timer.stage('base_offload', gpu=True):
            self.native.cpu()
            torch.cuda.empty_cache()
        with self.timer.stage('codec_and_joint_load'):
            self.model = rt.build_joint(self.a, self.saved, self.cfg, self.native, self.weights)
            self.load_counts.update(codec=1, joint=1)
        self.joint_step = int(self.saved['step'])
        self.saved = None
        gc.collect()
        rt.require(not any(p.requires_grad for p in self.model.parameters()), '推理模型必须冻结')
        select_runtime(self.a.runtime_root)

    def sample(self, target):
        a = self.target_args(target, 'ensemble_phase')
        model = self.model
        with self.timer.stage('ensemble_target', gpu=True, target=target['name']):
            with a.timing.stage('prepared_read'):
                payload, atoms = read_prepared(a.out, self.contracts[target['name']])
            identities = payload['identities']
            features = rt.move(payload['features'], self.device)
            features = update_input_feature_dict(model.native.relative_position_encoding.generate_relp(features))
            with torch.no_grad(), rt.amp(a):
                with a.timing.stage('online_trunk_to_gpu', gpu=True):
                    model.native.to(self.device)
                with a.timing.stage('online_context', gpu=True, cycles=model.config.n_cycle):
                    runner.reset_template_probe_rng(101)
                    online = model.online_context(features)
                rt.trace_inputs(a, 'online_context', online)
                with a.timing.stage('ensemble_setup', gpu=True):
                    model.native.cpu()
                    model.native.diffusion_module.to(self.device)
                    model.codec.to(self.device)
                    model.generator.to(self.device)
                    model.latent_mean = model.latent_mean.to(self.device)
                    model.latent_std = model.latent_std.to(self.device)
                    batch = rt.make_batch(features, payload['anchor'].to(self.device),
                                          tuple(v.to(self.device) for v in payload['c0']))
                proposals, predictions = [], []
                for seed in seed_list(a.seeds):
                    runner.reset_template_probe_rng(seed)
                    rt.trace_inputs(a, f'seed_{seed}_rng', dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device)))
                    with a.timing.stage('ensemble_seed', gpu=True, seed=seed, samples=a.samples_per_seed, chunk=a.sample_chunk_size):
                        outputs = runner.generate_samples_chunked(model, batch, online, reference_present=False,
                            samples=a.samples_per_seed, chunk_size=a.sample_chunk_size, sampler='sde_eta10')
                        out = {k:v.detach().cpu() for k,v in outputs.items()}
                    with a.timing.stage('seed_tensor_write', seed=seed):
                        torch.save(dict(out, seed=seed, atom_identities=identities), a.out / f'seed_{seed}.pt')
                    proposals.append(out['proposal'])
                    predictions.append(out['final'])
                    del outputs
            with a.timing.stage('ensemble_cif_write'):
                errors = dict(proposal=rt.write_coordinates(a.out / 'proposals.cif', atoms, identities, torch.cat(proposals)),
                    ensemble=rt.write_coordinates(a.out / 'ensemble.cif', atoms, identities, torch.cat(predictions)))
            total = len(seed_list(a.seeds))*a.samples_per_seed
            report = dict(status='inference_complete', target=target['name'], joint_step=self.joint_step,
                condition_contract=model.condition_contract,
                modality=target.get('modality','monomer'), n_chains=len(set(atoms.chain_id.tolist())),
                n_ligand_atoms=int(features['is_ligand'].sum()) if 'is_ligand' in features else None,
                checkpoint_sha256=self.hashes, input_identity=identities, n_atoms=len(atoms),
                n_tokens=len(features['token_index']), total_samples=total, seeds=seed_list(a.seeds),
                samples_per_seed=a.samples_per_seed, chunk_size=a.sample_chunk_size,
                effective_joint_config=asdict(model.config), sampler='sde_eta10', dtype=a.dtype, tf32=False,
                anchor_selection=self.contracts[target['name']]['anchor_selection'], msa_mode=target['msa_mode'],
                cif_roundtrip_max_error_A=errors, trainable_parameters_after_loading=0,
                dataset_frames_loaded=False, real_templates=False, code_sha256=self.code,
                runtime_imports={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
                    if n.split('.')[0] in ('protenix','runner','configs') and getattr(m,'__file__',None)},
                batch_peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(self.device)/2**30)
            with a.timing.stage('model_offload', gpu=True):
                model.cpu()
                del features, batch, online, payload
                torch.cuda.empty_cache()
            rt.save_json(a.out / 'run_metadata.json', report)
            rt.save_json(a.out / 'completion.json', report)
        times = a.timing.finish('inference_complete', total_samples=total)
        # 根计时保留子阶段来源；只汇总ensemble_seed，不把包裹它的ensemble_target重复相加。
        self.timer.data['stages'].extend(dict(e, target=target['name'], parent_stage='ensemble_target')
                                        for e in times['stages'] if e['stage'] == 'ensemble_seed')
        anchor_times = json.loads((a.out / 'anchor_phase/timing.json').read_text())
        self.results.append(dict(target=target['name'], total_samples=total,
            anchor_phase_seconds=anchor_times['wall_seconds'], ensemble_phase_seconds=times['wall_seconds'],
            ptx_anchor_seconds=sum(e['seconds'] for e in anchor_times['stages'] if e['stage']=='ptx_anchor'),
            ensemble_seconds=times['ensemble_seconds'], output=str(a.out)))

    def close(self):
        self.model = self.native = self.saved = None
        gc.collect()
        torch.cuda.empty_cache()
