"""发行版内自包含单体/PL/PP推理，无轨迹、无optimizer、无训练入口。"""
from __future__ import annotations

from dataclasses import asdict, replace
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from oridynamic.prepare import file_info
from oridynamic.weights_io import load_weights, PROFILE
from oridynamic.codec_condition import joint_model_class


def save_json(path, value):
    from oridynamic.runtime_utils import atomic_json
    atomic_json(Path(path), value)


def move(value, device):
    from oridynamic.runtime_utils import _move_to_device
    return _move_to_device(value, torch.device(device))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprints(value):
    """只读张量指纹，包含形状/dtype；兼容标量、空张量及BF16，不消耗RNG。"""
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest())
    if isinstance(value, dict):
        return {str(k): fingerprints(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [fingerprints(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"未支持的指纹对象：{type(value).__name__}")


def trace_inputs(args, name, value):
    if not getattr(args, "trace_inputs", False):
        return
    with args.timing.stage('input_fingerprints', part=name):
        path = args.out / 'input_fingerprints.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        record[name] = fingerprints(value)
        save_json(path, record)


def amp(args):
    from contextlib import nullcontext
    return (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
            if args.dtype == "bf16" and not args.check else nullcontext())


def load_parent(weights):
    saved = load_weights(weights["joint"], "joint")
    meta = saved["metadata"]
    joint_model_class(meta["condition_contract"])  # 拒绝未知合同，不按文件名猜测。
    require(meta["codec_identity"]["coordinate_frame"] == "reference_canonical", "需要旧canonical codec")
    require(not meta["stageb_model_config"]["use_ref"], "首版只支持已验证的ref-off Joint；用户结构是decoder anchor")
    require(meta["online_template_execution_contract"] == "joint_online_native_null_template_v1",
            "首版要求已验证的原生空模板Joint")
    hashes = {key: file_info(path)["sha256"] for key, path in weights.items()}
    require(hashes["native"] == meta["codec_identity"]["trunk_checkpoint_sha256"], "base PTX与codec不匹配")
    require(hashes["codec"] == meta["codec_identity"]["vae_sha256"], "AE与Joint不匹配")
    return saved, hashes


def make_config(args, saved):
    from ml_collections import ConfigDict
    from oridynamic.runtime_utils import _stable_json_sha256
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    cfg = ConfigDict(json.loads((config_dir / "inference_defaults.json").read_text()))
    # 导出时核验原始配置；资源包只保留明确的推理profile，不携带训练命令。
    require(saved["metadata"]["native_inference_profile"] == PROFILE, "未知原生推理配置")
    require(cfg.data.msa.max_size.test == 512 and cfg.data.msa.sample_cutoff.test == 512
            and cfg.model.diffusion_module.blocks_per_ckpt == 3, "发布配置偏离权重绑定profile")
    base_diffusion = json.loads((config_dir / "base_diffusion_signature.json").read_text())
    require(_stable_json_sha256(base_diffusion) == saved["metadata"]["codec_identity"]["trunk_diffusion_config_sha256"],
            "当前原生架构配置与冻结codec来源不同")
    actual_diffusion = cfg.model.diffusion_module.to_dict()
    actual_diffusion["blocks_per_ckpt"] = base_diffusion["blocks_per_ckpt"]
    require(actual_diffusion == base_diffusion, "Joint配置包含非checkpoint分组的diffusion架构修改")
    cfg.input_json_path = str(args.out / "input.json")
    cfg.dump_dir = str(args.out / "native")
    cfg.use_template = False
    cfg.data.template.enable_prot_template = False
    cfg.data.template.fetch_remote = False
    cfg.use_msa = True  # 单序列模式也走原生query MSA，而不是修改模型配置。
    cfg.esm.enable = False
    cfg.enable_tf32 = False
    cfg.dtype = args.dtype
    cfg.infer_setting.dynamic_chunk_size = False
    cfg.infer_setting.chunk_size = args.attention_chunk_size
    cfg.triangle_attention = args.attention_backend
    cfg.triangle_multiplicative = args.attention_backend
    cfg.model.N_cycle = 10  # 仅原生anchor；C0和在线trunk分别显式4cycle。
    cfg.model.N_model_seed = 1
    cfg.sample_diffusion.N_sample = 1
    require(cfg.sample_diffusion.N_step == 200, "base PTX原生schedule已改变")
    require(all(cfg.sample_diffusion[k] == v for k, v in
            dict(gamma0=.8, gamma_min=1., noise_scale_lambda=1.003, step_scale_eta=1.5).items()),
            "原生anchor SDE协议已改变")
    return cfg


def prepare_features(args, cfg, native_input):
    from runner.msa_search import update_infer_json
    from protenix.data.inference.infer_dataloader import InferenceDataset
    from oridynamic import runtime_utils as runner
    from oridynamic.runtime_utils import build_reference_identities
    native_input = copy.deepcopy(native_input)
    require(len(native_input) == 1, '每次特征处理须为一个完整体系')
    protein_entities = [e['proteinChain'] for e in native_input[0]['sequences'] if 'proteinChain' in e]
    require(bool(protein_entities), '至少需要一条蛋白链')
    if args.single_sequence:
        for i, protein in enumerate(protein_entities):
            if not protein.get('unpairedMsaPath'):
                msa_path = args.out / ('query.a3m' if len(protein_entities) == 1 else f'query_entity_{i}.a3m')
                msa_path.write_text(">query\n" + protein["sequence"] + "\n")
                protein["unpairedMsaPath"] = str(msa_path)
    save_json(args.out / "input.json", native_input)
    if not all(p.get('unpairedMsaPath') for p in protein_entities):
        require(not hasattr(args, 'input_json'), '批量入口缺少逐链MSA，禁止隐式在线搜索')
        print(f"[MSA] 使用{args.msa_server_mode}服务；将发送输入序列", flush=True)
        with args.timing.stage('msa_online'):
            cfg.input_json_path, _ = update_infer_json(str(args.out / "input.json"), str(args.out / "msa"),
                                                       use_msa=True, mode=args.msa_server_mode)
    with args.timing.stage('native_features'):
        runner.reset_template_probe_rng(101)
        ds = InferenceDataset(cfg)
        data, atoms, error = ds[0]
    require(not error, f"原生JSON特征处理失败：{error}")
    features = data["input_feature_dict"]
    runner.validate_null_templates(features)
    validate_complex_features(features, atoms, native_input[0])
    identities = build_reference_identities(atoms)
    return features, atoms, identities


def validate_complex_features(features, atoms, sample):
    """在原生featurizer后检查整条链/配体均保留；不重排、不重新生成化学特征。"""
    require(bool((features['is_protein'].bool() | features['is_ligand'].bool()).all()),
            '本入口只验收蛋白及配体原子')
    expected = sum(next(iter(e.values()))['count'] for e in sample['sequences'])
    require(len(set(atoms.chain_id.tolist())) == expected and len(torch.unique(features['asym_id'])) == expected,
            '原生特征的链数与输入不符，拒绝丢链/裁剪')
    declared = [c for e in sample['sequences'] for c in next(iter(e.values())).get('id', [])]
    if declared:
        require(set(declared) == set(atoms.chain_id.tolist()), '链ID与原生原子轴不符')
    ligand_ids = [c for e in sample['sequences'] if 'ligand' in e for c in e['ligand'].get('id', [])]
    for chain in ligand_ids:
        indices = torch.from_numpy(np.flatnonzero(atoms.chain_id == chain))
        require(len(indices) > 0 and bool(features['is_ligand'][indices].bool().all()), '配体未完整进入原生特征')
    require(len(atoms) == len(features['atom_to_token_idx']) and len(atoms) == len(features['ref_pos']),
            '原子坐标/化学特征/token映射轴不一致')


def map_structure(path, expected_atoms, chain=None, chain_map=None):
    """同序列按残基顺序建映射；不拟合坐标，不补原子，不从GT选择。"""
    import biotite.structure as struc
    from biotite.structure.io import pdb, pdbx
    from protenix.data.ensemble_schema import build_atom_identity_mapping
    if Path(path).suffix.lower() == ".pdb":
        atoms = pdb.PDBFile.read(path).get_structure(model=1, altloc="occupancy")
    else:
        atoms = pdbx.get_structure(pdbx.CIFFile.read(path), model=1, altloc="occupancy", use_author_fields=True)
    heavy = ~np.isin(atoms.element, ['H', 'D'])
    complex_input = len(set(expected_atoms.chain_id.tolist())) > 1 or not bool(struc.filter_amino_acids(expected_atoms).all())
    if complex_input or chain_map is not None:
        require(chain is None, '复合物请使用structure_chains，不使用单链structure_chain')
        return map_complex_atoms(atoms[heavy], expected_atoms, chain_map)
    atoms = atoms[struc.filter_amino_acids(atoms) & heavy]
    chains = set(atoms.chain_id.tolist())
    require(chain in chains if chain is not None else len(chains) == 1,
            "初始结构须明确单条蛋白链；多链时指定--structure-chain")
    atoms = atoms[atoms.chain_id == (chain if chain is not None else next(iter(chains)))]

    def keys(array):
        starts = struc.get_residue_starts(array, add_exclusive_stop=True)
        names, ids = [], []
        for ordinal, (first, last) in enumerate(zip(starts[:-1], starts[1:]), 1):
            names.append(str(array.res_name[first]))
            ids.extend(dict(asym_id="A", residue_index=ordinal, component_id=str(array.res_name[i]),
                            atom_name=str(array.atom_name[i]), insertion_code="") for i in range(first, last))
        return names, ids

    expected_names, expected_ids = keys(expected_atoms)
    actual_names, actual_ids = keys(atoms)
    require(expected_names == actual_names, "初始结构残基序列/长度与FASTA不同；首版不猜测缺失残基")
    mapping = build_atom_identity_mapping(expected_ids, actual_ids, require_all_reference=True,
                                          require_all_target=False)
    indices = mapping.reference_to_target.numpy()
    require(np.array_equal(expected_atoms.element, atoms.element[indices]), "初始结构元素与原生原子身份不同")
    xyz = torch.from_numpy(atoms.coord[indices].copy()).float()
    require(bool(torch.isfinite(xyz).all()), "初始结构含非有限坐标")
    return xyz


def map_complex_atoms(actual, expected, chain_map=None):
    """全复合物同一坐标系；逐链按残基顺序/组分/原子名严格映射，绝不单独拟合配体。"""
    import biotite.structure as struc
    from protenix.data.ensemble_schema import build_atom_identity_mapping
    chains = list(dict.fromkeys(expected.chain_id.tolist()))
    chain_map = chain_map if chain_map is not None else {c:c for c in chains}
    require(set(chain_map) == set(chains) and len(set(chain_map.values())) == len(chains),
            'structure_chains必须完整且一对一，不猜测同源链置换')
    indices = np.full(len(expected), -1, dtype=np.int64)
    for chain in chains:
        ei = np.flatnonzero(expected.chain_id == chain)
        ai = np.flatnonzero(actual.chain_id == chain_map[chain])
        require(len(ai) > 0, f'anchor缺少链 {chain_map[chain]}（输入 {chain}）')
        e, a = expected[ei], actual[ai]
        def keys(array):
            names, ids = [], []
            starts = struc.get_residue_starts(array, add_exclusive_stop=True)
            for ordinal, (first,last) in enumerate(zip(starts[:-1],starts[1:]),1):
                names.append(str(array.res_name[first]))
                ids.extend(dict(asym_id=chain, residue_index=ordinal, component_id=str(array.res_name[i]),
                                atom_name=str(array.atom_name[i]), insertion_code='') for i in range(first,last))
            return names, ids
        en, ek = keys(e); an, ak = keys(a)
        require(en == an, f'anchor链{chain}: 残基/配体CCD组分或长度不符；不补缺失残基或猜测配体原子名')
        mapping = build_atom_identity_mapping(ek, ak, require_all_reference=True, require_all_target=False)
        indices[ei] = ai[mapping.reference_to_target.numpy()]
    require(bool((indices >= 0).all()) and len(set(indices.tolist())) == len(expected), 'anchor映射缺失或重复')
    require(np.array_equal(expected.element, actual.element[indices]), 'anchor映射的元素不同')
    xyz = torch.from_numpy(actual.coord[indices].copy()).float()
    require(bool(torch.isfinite(xyz).all()), 'anchor包含非有限坐标')
    return xyz


def choose_anchor(coordinates, scores, n_atoms=None):
    require(len(coordinates) == len(scores) and len(scores) > 0, "候选数和置信度不一致")
    from protenix.model.coordinate_frame import build_reference_canonical_frame
    expected = (n_atoms if n_atoms is not None else len(coordinates[0]), 3)
    valid = [i for i, (xyz, score) in enumerate(zip(coordinates, scores))
             if tuple(xyz.shape) == expected and bool(torch.isfinite(xyz).all()) and np.isfinite(score)]
    usable = []
    for index in valid:
        try:
            build_reference_canonical_frame(coordinates[index], torch.ones(expected[0], dtype=torch.bool))
            usable.append(index)
        except ValueError:
            pass
    valid = usable
    require(bool(valid), "没有完整、有限且可建立canonical坐标系的PTX候选；不回退到GT或噪声anchor")
    return min(valid, key=lambda i: (-scores[i], i))


def make_batch(features, anchor, context):
    from protenix.model.coordinate_frame import canonicalize_batch
    from oridynamic.runtime_utils import reference_token_centroids
    require(anchor.shape == (len(features["atom_to_token_idx"]), 3), "anchor原子轴不同")
    mask = torch.ones(len(anchor), dtype=torch.bool, device=anchor.device)
    teacher = dict(reference_coordinate=anchor, coordinate_mask=mask, reference_coordinate_mask=mask,
                   s_inputs=context[0], s_trunk=context[1], z_trunk=context[2], input_feature_dict=features)
    teacher, frame = canonicalize_batch(teacher, "reference_canonical")
    reference_token, token_mask = reference_token_centroids(teacher["reference_coordinate"], mask,
                                  features["atom_to_token_idx"], len(context[0]))
    return dict(features=features, codec_context=teacher, canonical_frame=frame,
                atom_mask=mask, reference_token=reference_token.to(anchor.device),
                token_mask=token_mask.to(anchor.device))


def build_joint(args, saved, cfg, native, weights):
    from oridynamic import runtime_utils as runner
    from protenix.model.lora import inject_lora
    from protenix.model.modules.joint_ensemble import JointConfig, JointEnsembleModel
    from oridynamic.runtime_utils import load_vae_checkpoint, _build_denoiser
    meta = saved["metadata"]
    diffusion_trainable = {n for n, p in native.diffusion_module.named_parameters() if p.requires_grad}
    native.requires_grad_(False)
    for name, param in native.diffusion_module.named_parameters():
        param.requires_grad_(name in diffusion_trainable)
    lora = meta["lora"]
    require(set(lora["modules"]) <= {"msa_module", "pairformer_stack", "input_embedder"}, "未知LoRA组")
    for name in lora["modules"]:
        require(inject_lora(getattr(native, name), lora["r"], lora["alpha"], dropout=0) > 0, "未注入LoRA")
    summary = {key: int(cfg.model.diffusion_module[key]) for key in ("c_s_inputs", "c_s", "c_z")}
    codec, _ = load_vae_checkpoint(weights["codec"], summary, torch.device("cpu"))
    generator = _build_denoiser(meta["stageb_backbone"], meta["stageb_model_config"])
    config = JointConfig.from_checkpoint(saved["joint_config"])
    state = saved["trainable_state"]
    model_class = joint_model_class(meta["condition_contract"])
    model = model_class(native, generator, codec, state["latent_mean"], state["latent_std"], config)
    model.condition_contract = meta["condition_contract"]
    model.config = replace(config, activation_checkpoint=False, chunk_size=args.attention_chunk_size)
    require(set(state) == runner.checkpoint_state_names(model), "Joint参数/buffer集合不符")
    current = model.state_dict()
    for name, value in state.items():
        require(current[name].shape == value.shape and current[name].dtype == value.dtype, f"张量不匹配：{name}")
    current.update(state)
    model.load_state_dict(current, strict=True)
    runner.configure_sample_template(model, SimpleNamespace(sample_online_template=None),
                                     meta["online_template_execution_contract"])
    require(runner.native_null_template_settings(model) == meta["online_template_settings"], "空模板通路改变")
    model.codec_identity = meta["codec_identity"]
    return model.eval().requires_grad_(False)


def write_coordinates(path, atoms, identities, coordinates):
    import biotite.structure as struc
    from oridynamic.runtime_utils import write_cif
    values = coordinates.detach().cpu().float().numpy()
    require(values.ndim == 3 and values.shape[1:] == (len(atoms), 3) and np.isfinite(values).all(), "输出坐标无效")
    stack = struc.stack([atoms.copy() for _ in range(len(values))])
    stack.coord = values
    stack.set_annotation("occupancy", np.ones(len(atoms)))
    stack.set_annotation("b_factor", np.zeros(len(atoms)))
    return write_cif(path, stack, identities)


def run(args, paths, weights, native_input):
    from oridynamic import runtime_utils as runner
    from protenix.model.protenix import Protenix, update_input_feature_dict
    from oridynamic.predict import select_runtime
    select_runtime(args.runtime_root)  # 再验已导入模块的真实来源。
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cpu" if args.check else args.device)
    timer = args.timing
    if not args.check:
        require(device.type == "cuda" and torch.cuda.is_available(), "推理请在PTX兼容CUDA环境执行；CPU仅--check")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        timer.configure_cuda(lambda: torch.cuda.synchronize(device), device=str(device),
                             gpu_name=torch.cuda.get_device_name(device))
    start = time.monotonic()
    print("[权重] 检查base/AE/Joint身份；不读取训练数据", flush=True)
    with timer.stage('joint_load_and_weight_hashes'):
        saved, hashes = load_parent(weights)
        cfg = make_config(args, saved)
    features_cpu, atoms, identities = prepare_features(args, cfg, native_input)
    trace_inputs(args, 'features', features_cpu)
    with timer.stage('native_model_load'):
        runner.reset_template_probe_rng(101)
        native = Protenix(cfg)
        base = load_weights(weights["native"], "native")
        native.load_state_dict({k.removeprefix("module."): v for k, v in base.get("model", base).items()}, strict=True)
        del base
        native.eval()  # 此处暂保留requires_grad标志，后续严格复原Joint保存的参数集合。
    user_anchor = None
    if args.initial_structure:
        with timer.stage('user_anchor_map'):
            user_anchor = map_structure(args.initial_structure, atoms, args.structure_chain)
    report = dict(schema="oridynamic_sequence_inference_v1", joint_step=int(saved["step"]),
        condition_contract=saved["metadata"]["condition_contract"],
        checkpoint_sha256=hashes, runtime_root=str(args.runtime_root), n_atoms=len(atoms),
        n_tokens=len(features_cpu["token_index"]), msa_mode="local" if getattr(args, "msa_local_db", None) else "single_sequence" if args.single_sequence else "provided" if args.msa_a3m else "server",
        msa_search_report=file_info(args.out / "msa/search.json") if getattr(args, "msa_local_db", None) else None,
        features_from="native_InferenceDataset_no_trajectory", dataset_frames_loaded=False,
        anchor_source="user_structure" if user_anchor is not None else "ptx_predict",
        seeds=[int(x) for x in args.seeds.split(",")], samples_per_seed=args.samples_per_seed,
        chunk_size=args.sample_chunk_size, feature_seed=101, codec_cycles=4, anchor_cycles=10,
        online_cycles=saved["joint_config"]["n_cycle"], real_templates=False, tf32=False,
        trace_inputs=bool(getattr(args, 'trace_inputs', False)),
        dtype=args.dtype, attention_backend=args.attention_backend,
        native_sampling=cfg.sample_diffusion.to_dict(), saved_joint_config=saved["joint_config"],
        input_identity=identities, torch_version=torch.__version__, sampler="sde_eta10",
        initial_structure=None if args.initial_structure is None else file_info(args.initial_structure),
        code_sha256={name:file_info(args.runtime_root/name)["sha256"] for name in (
            "oridynamic/runtime_utils.py", "oridynamic/codec_condition.py", "protenix/model/modules/joint_ensemble.py",
            "protenix/data/inference/infer_dataloader.py", "runner/msa_search.py")},
        entry_sha256={name:file_info(Path(__file__).parent/name)["sha256"]
                      for name in ("predict.py", "inference_runtime.py", "prepare.py", "local_msa.py", "weights_io.py", "inference_timing.py")})
    if not args.check:
        with timer.stage('native_to_gpu', gpu=True):
            features = move(features_cpu, device)
            native.to(device)
            features = update_input_feature_dict(native.relative_position_encoding.generate_relp(features))
        with torch.no_grad(), amp(args):
            with timer.stage('base_context', gpu=True, cycles=4):
                runner.reset_template_probe_rng(101)
                c0 = native.get_pairformer_output(features, 4, inplace_safe=False, chunk_size=args.attention_chunk_size)
                c0 = tuple(x.detach().cpu() for x in c0)
            trace_inputs(args, 'base_context', c0)
            if user_anchor is None:
                candidates, scores = [], []
                for i in range(args.anchor_candidates):
                    runner.reset_template_probe_rng(args.anchor_seed + i)
                    print(f"[PTX anchor] {i+1}/{args.anchor_candidates}；纯噪声SDE200", flush=True)
                    with timer.stage('ptx_anchor', gpu=True, candidate=i+1, seed=args.anchor_seed+i):
                        pred = native(input_feature_dict=copy.deepcopy(features), label_full_dict=None,
                                      label_dict=None, mode="inference", mc_dropout_apply_rate=0., disable_inplace=True)[0]
                        candidates.append(pred["coordinate"][0].detach().cpu().float())
                        scores.append(float(pred["summary_confidence"][0]["ranking_score"]))
                        del pred
                selected = choose_anchor(candidates, scores, len(atoms))
                user_anchor = candidates[selected]
                report["anchor_selection"] = dict(policy="native_ranking_score_top1",
                    scores=[value if np.isfinite(value) else None for value in scores],
                    seeds=list(range(args.anchor_seed, args.anchor_seed + args.anchor_candidates)), selected=selected)
                del candidates
        with timer.stage('native_offload', gpu=True):
            native.cpu()
            del features
            torch.cuda.empty_cache()
    print("[Joint] 严格恢复DiT/diffusion/LoRA；不创建optimizer", flush=True)
    with timer.stage('codec_and_joint_load'):
        model = build_joint(args, saved, cfg, native, weights)
    select_runtime(args.runtime_root)
    report["runtime_imports"] = {name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name.split(".")[0] in {"protenix", "runner", "configs"}
        and getattr(module, "__file__", None)}
    report["trainable_parameters_after_loading"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(report["trainable_parameters_after_loading"] == 0, "推理模型未完全冻结")
    report["effective_joint_config"] = asdict(model.config)
    report["online_template"] = runner.native_null_template_settings(model)
    report["parameter_state_count"] = len(saved["trainable_state"])
    del saved
    gc.collect()
    save_json(args.out / "resolved_config.json", cfg.to_dict())
    save_json(args.out / "run_metadata.json", report)
    if args.check:
        return dict(status="cpu_features_and_strict_weights_passed", model_forward=False,
                    n_atoms=len(atoms), n_tokens=report["n_tokens"], elapsed_seconds=time.monotonic()-start)
    features = move(features_cpu, device)
    features = update_input_feature_dict(model.native.relative_position_encoding.generate_relp(features))
    with torch.no_grad(), amp(args):
        print("[上下文] 计算在线LoRA s/z；随后将不用的trunk移回CPU", flush=True)
        with timer.stage('online_trunk_to_gpu', gpu=True):
            model.native.to(device)
        with timer.stage('online_context', gpu=True, cycles=model.config.n_cycle):
            runner.reset_template_probe_rng(101)
            online = model.online_context(features)
        trace_inputs(args, 'online_context', online)
        with timer.stage('ensemble_setup', gpu=True):
            model.native.cpu()
            model.native.diffusion_module.to(device)
            model.codec.to(device)
            model.generator.to(device)
            model.latent_mean = model.latent_mean.to(device)
            model.latent_std = model.latent_std.to(device)
            batch = make_batch(features, user_anchor.to(device), tuple(x.to(device) for x in c0))
        trace_inputs(args, 'anchor', user_anchor)
        with timer.stage('anchor_cif_write'):
            write_coordinates(args.out / "anchor.cif", atoms, identities, user_anchor[None])
        proposals, predictions = [], []
        for seed in report["seeds"]:
            runner.reset_template_probe_rng(seed)
            trace_inputs(args, f'seed_{seed}_rng', dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device)))
            print(f"[ensemble] seed={seed} 样本={args.samples_per_seed} chunk={args.sample_chunk_size}", flush=True)
            with timer.stage('ensemble_seed', gpu=True, seed=seed, samples=args.samples_per_seed,
                             chunk=args.sample_chunk_size):
                outputs = runner.generate_samples_chunked(model, batch, online, reference_present=False,
                    samples=args.samples_per_seed, chunk_size=args.sample_chunk_size, sampler="sde_eta10")
                out = {k: v.detach().cpu() for k, v in outputs.items()}
            with timer.stage('seed_tensor_write', seed=seed):
                torch.save(dict(out, seed=seed, atom_identities=identities), args.out / f"seed_{seed}.pt")
            proposals.append(out["proposal"])
            predictions.append(out["final"])
            del outputs
    with timer.stage('ensemble_cif_write'):
        errors = {"proposal": write_coordinates(args.out / "proposals.cif", atoms, identities, torch.cat(proposals)),
                  "ensemble": write_coordinates(args.out / "ensemble.cif", atoms, identities, torch.cat(predictions))}
    report.update(status="inference_complete", total_samples=len(report["seeds"])*args.samples_per_seed,
                  cif_roundtrip_max_error_A=errors, elapsed_seconds=time.monotonic()-start,
                  max_cuda_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                  max_cuda_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
    save_json(args.out / "run_metadata.json", report)
    return report
