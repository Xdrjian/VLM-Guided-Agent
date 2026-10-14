"""第 4 层（闭环）自测：``python tests/test_loop.py``。

本文件刻意**不依赖 torch、不依赖 opencv、不依赖任何网络**：
外层循环的渲染与求解都以注入点接入，这里用假的渲染器/求解器驱动全流程，
因此可以在没有 GPU、没有 API Key 的机器上验证循环逻辑本身。

覆盖：
- VLMScore 分档校验、归一化、加权总分的单调性
- 证据包落盘/回读的往返一致性
- History 的接受/回滚语义、"调参基底永远是历史最佳"、终止条件
- prompt 渲染包含关键要素，且不泄漏加权总分给 LLM
- JSON 抽取对三种返回形态的鲁棒性
- 手动客户端的挂起-续跑（断点幂等）
- 端到端假循环：第0轮健康检查、多次打分取中位数、噪声极差统计
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionedit.clients import (                                   # noqa: E402
    GEMINI_VIDEO_FPS, ApiVLMClient, GeminiLLMClient, GeminiVLMClient,
    ManualLLMClient, ManualVLMClient, PendingManualInput,
)
from motionedit.history import History, StopReason, TerminationConfig  # noqa: E402
from motionedit.outer_loop import (                                # noqa: E402
    LoopConfig, OuterLoop, RenderOutput, SolveOutput,
)
from motionedit.params import DEFAULT_PARAMS, GaitParams           # noqa: E402
from motionedit.prompts import (                                   # noqa: E402
    GridSpec, VideoSpec, extract_json, render_llm_prompt, render_vlm_prompt,
    render_vlm_prompt_round0,
)
from motionedit.records import EvidenceStore, RawResponse, RoundRecord, VLMScore  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


def _score(w=3, t=3, s=3, f=3, n=7) -> VLMScore:
    return VLMScore(weight=w, time=t, space=s, flow=f, naturalness=n,
                    comments={k: "x" for k in ("weight", "time", "space", "flow", "naturalness")})


# --------------------------------------------------------------------------- #
def test_score():
    print("\n[1] VLMScore")
    lo = VLMScore(1, 1, 1, 1, 1)
    hi = VLMScore(5, 5, 5, 5, 10)
    check("最低分归一化为 0", abs(lo.total()) < 1e-9, f"{lo.total():.6f}")
    check("最高分归一化为 1", abs(hi.total() - 1.0) < 1e-9, f"{hi.total():.6f}")
    check("总分单调递增", _score(w=2).total() < _score(w=4).total())

    # Weight 权重应显著大于其它 Effort 维度
    base = _score()
    dw = _score(w=5).total() - base.total()
    dt = _score(t=5).total() - base.total()
    check("Weight 维度权重 > Time 维度", dw > dt + 1e-9, f"ΔW={dw:.4f} ΔT={dt:.4f}")

    # 自然度权重 = 四项 Effort 之和 → 自然度满档的贡献等于四项 Effort 满档之和
    all_eff = VLMScore(5, 5, 5, 5, 1).total()
    all_nat = VLMScore(1, 1, 1, 1, 10).total()
    check("自然度权重 = 四项 Effort 权重之和", abs(all_eff - all_nat) < 1e-9,
          f"{all_eff:.4f} vs {all_nat:.4f}")

    bad = VLMScore(weight=7, time=3, space=3, flow=3, naturalness=7)
    check("越界分数被 validate 捕获", len(bad.validate()) == 1, str(bad.validate()))
    check("自然度 10 分合法", len(VLMScore(3, 3, 3, 3, 10).validate()) == 0)
    check("from_dict 容忍多余字段与字符串数字",
          VLMScore.from_dict({"weight": "4", "time": 3, "space": 3, "flow": 3,
                              "naturalness": 8, "extra": 1}).weight == 4)


# --------------------------------------------------------------------------- #
def test_records(tmp: Path):
    print("\n[2] 证据包 EvidenceStore")
    store = EvidenceStore(tmp / "run_a")
    rec = RoundRecord(round_index=3, params=GaitParams(knee_flex_scale=1.3),
                      features={"knee_peak": 122.7}, score=_score(w=4),
                      llm_reasoning="加深屈膝", solver_status="converged")
    store.save_record(rec)
    back = store.load_record(3)
    check("记录往返：参数一致", back.params.knee_flex_scale == 1.3)
    check("记录往返：分数一致", back.score.to_dict() == rec.score.to_dict())
    check("记录往返：总分一致", abs(back.total_score - rec.total_score) < 1e-12)
    check("list_rounds 能发现已落盘轮次", store.list_rounds() == [3])

    resp = RawResponse(text='{"a":1}', body={"model": "m-2026-01-01"},
                       model="m-2026-01-01", request={"temperature": 0.0})
    store.save_response(3, "vlm", resp)
    saved = store.read_json(store.round_dir(3) / "vlm_response.json")
    check("原始返回体全文落盘", saved["text"] == '{"a":1}' and saved["model"] == "m-2026-01-01")
    check("请求参数单独落盘", store.read_json(store.round_dir(3) / "vlm_request.json")
          ["temperature"] == 0.0)

    p = store.images_dir(3) / "cam0.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    check("相对路径跨机器可读", store.rel(p) == "rounds/round_003/images/cam0.png",
          store.rel(p))


# --------------------------------------------------------------------------- #
def test_history(tmp: Path):
    print("\n[3] History 接受/回滚/终止")
    h = History(TerminationConfig(patience=2, score_target=0.99, max_rounds=10))
    r0 = h.add(RoundRecord(0, DEFAULT_PARAMS, score=_score(w=2, n=9), is_reference=True))
    check("第0轮成为初始最佳", r0.accepted and h.best.round_index == 0)

    r1 = h.add(RoundRecord(1, GaitParams(knee_flex_scale=1.2), score=_score(w=4, n=8)))
    check("更高分被接受为新最佳", r1.accepted and h.best.round_index == 1)

    r2 = h.add(RoundRecord(2, GaitParams(knee_flex_scale=1.9), score=_score(w=1, n=5)))
    check("更低分被回滚", r2.rolled_back and not r2.accepted)
    check("回滚后最佳不变（调参基底仍是历史最佳）", h.best.round_index == 1)
    check("回滚记录仍被保留（反面参考）", len(h.records) == 3 and h.records[2].score is not None)
    check("停滞计数递增", h.stall_count() == 1)

    check("recent 排除历史最佳本身", [r.round_index for r in h.recent(2)] == [2],
          str([r.round_index for r in h.recent(2)]))
    check("recent 不含第0轮参考", all(not r.is_reference for r in h.recent(5)))

    h.add(RoundRecord(3, GaitParams(), score=_score(w=1, n=4)))
    check("连续停滞达 patience 触发终止", h.should_stop() == StopReason.NO_IMPROVEMENT,
          str(h.stop_detail))

    h2 = History(TerminationConfig(patience=9, score_target=0.5, max_rounds=99))
    h2.add(RoundRecord(0, DEFAULT_PARAMS, score=VLMScore(5, 5, 5, 5, 10)))
    check("达到分数上限触发终止", h2.should_stop() == StopReason.SCORE_TARGET)

    h3 = History(TerminationConfig(patience=99, score_target=1.1, max_rounds=2))
    h3.add(RoundRecord(0, DEFAULT_PARAMS, score=_score()))
    h3.add(RoundRecord(1, DEFAULT_PARAMS, score=_score(w=4)))
    check("达到最大轮数触发终止", h3.should_stop() == StopReason.MAX_ROUNDS)

    # 第 0 轮健康检查
    h4 = History(TerminationConfig(ref_naturalness_floor=8))
    h4.add(RoundRecord(0, DEFAULT_PARAMS, score=_score(n=5), is_reference=True))
    check("A_ref 自然度偏低触发 VLM 可靠性告警", len(h4.check_reference_health()) == 1)
    h5 = History(TerminationConfig(ref_naturalness_floor=8))
    h5.add(RoundRecord(0, DEFAULT_PARAMS, score=_score(n=9), is_reference=True))
    check("A_ref 自然度正常则无告警", len(h5.check_reference_health()) == 0)

    # 轮次编号必须连续
    try:
        History().add(RoundRecord(5, DEFAULT_PARAMS, score=_score()))
        check("轮次编号不连续应报错", False)
    except ValueError:
        check("轮次编号不连续应报错", True)

    # 持久化往返
    p = h.save(tmp / "hist.json")
    h6 = History.load(p)
    check("History 往返：最佳轮次一致", h6.best.round_index == h.best.round_index)
    check("History 往返：停滞计数一致", h6.stall_count() == h.stall_count())
    check("History 往返：终止阈值一致", h6.cfg.patience == h.cfg.patience)


# --------------------------------------------------------------------------- #
def test_prompts():
    print("\n[4] Prompt 模板")
    grid = GridSpec()
    check("默认宫格为 2×4 = 8 帧", grid.frames_per_camera == 8)
    check("默认 3 个机位", len(grid.cameras) == 3)
    check("宫格像素随 cell_px 推导", GridSpec(cell_px=768).grid_px == (3072, 1536),
          str(GridSpec(cell_px=768).grid_px))

    best = RoundRecord(1, GaitParams(knee_flex_scale=1.25), features={"knee_peak": 120.0},
                       score=_score(w=4), accepted=True)
    p = render_vlm_prompt(2, 30.0, best, grid)
    check("VLM prompt 声明左右列含义", "历史最佳解" in p and "本轮候选解" in p)
    check("VLM prompt 含目标重量", "30kg" in p)
    check("VLM prompt 含 BARS 锚点", "完全看不出负重痕迹" in p)
    check("VLM prompt 明确排除物理合理性评判", "不要评估物理合理性" in p)
    check("VLM prompt 要求 JSON 输出", '"naturalness": int' in p)
    check("VLM prompt 锚点取历史最佳而非上一轮", "第 1 轮" in p)

    p0 = render_vlm_prompt_round0(30.0, grid)
    check("第0轮 prompt 声明这是动捕真值", "ground truth" in p0)
    check("第0轮 prompt 仍要求如实评 Weight", "如实独立判断" in p0)
    check("第0轮 prompt 不含左右列对比说明", "本轮候选解" not in p0)

    rolled = RoundRecord(2, GaitParams(knee_flex_scale=1.9), score=_score(w=1),
                         rolled_back=True, features={"knee_peak": 150.0})
    lp = render_llm_prompt(3, 30.0, best, [rolled], last_features={"knee_peak": 150.0})
    check("LLM prompt 声明基底为历史最佳", "本轮参数修改的基底" in lp)
    check("LLM prompt 含回滚记录作反面参考", "已回滚" in lp)
    check("LLM prompt 要求输出绝对值", "绝对数值" in lp)
    check("LLM prompt 列出全部 7 个参数", all(k in lp for k in DEFAULT_PARAMS.to_dict()))
    check("LLM prompt 含取值范围", "[0.5, 2.0]" in lp)
    check("LLM prompt 不泄漏加权总分", "total" not in lp and "加权" not in lp)
    check("LLM prompt 含求解器执行反馈", "求解器执行反馈" in lp)
    check("LLM prompt 使用中性第三人称（不含'你上轮'）", "你上轮" not in lp and "你的决定" not in lp)

    lp0 = render_llm_prompt(1, 30.0, None)
    check("无历史最佳时给出保守初始化指引", "尚无已接受的历史最佳解" in lp0)


# --------------------------------------------------------------------------- #
def test_extract_json():
    print("\n[5] JSON 抽取鲁棒性")
    d = {"weight": 3, "comments": {"weight": "还行"}}
    raw = json.dumps(d, ensure_ascii=False)
    check("纯 JSON", extract_json(raw) == d)
    check("代码块包裹", extract_json(f"```json\n{raw}\n```") == d)
    check("前后有散文", extract_json(f"好的，我的评分如下：\n{raw}\n希望有帮助。") == d)
    check("含嵌套与转义引号",
          extract_json('前言 {"a": {"b": "他说\\"好\\""}, "c": 1} 后记')["c"] == 1)
    for bad, why in [("完全没有花括号", "无 JSON"), ('{"a": 1', "未闭合"), ('{"a": }', "语法错")]:
        try:
            extract_json(bad)
            check(f"非法输入应报错（{why}）", False)
        except ValueError:
            check(f"非法输入应报错（{why}）", True)


# --------------------------------------------------------------------------- #
def test_clients(tmp: Path):
    print("\n[6] 客户端：手动挂起与版本纪律")
    pp, ap = tmp / "c" / "p.txt", tmp / "c" / "a.json"
    cli = ManualVLMClient(pp, ap)
    try:
        cli.call("SYS", "USER", [tmp / "img0.png"])
        check("答案缺失时应挂起", False)
    except PendingManualInput as e:
        check("答案缺失时应挂起", True)
        check("挂起时 prompt 已落盘", pp.exists())
        check("挂起指引含答案文件路径", str(ap) in str(e))
        check("挂起指引含图像清单", "img0.png" in str(e))
    txt = pp.read_text(encoding="utf-8")
    check("落盘 prompt 含 system 与 user", "SYS" in txt and "USER" in txt)

    ap.write_text('{"weight": 4}', encoding="utf-8")
    resp = cli.call("SYS", "USER")
    check("回填后可续跑", resp.text == '{"weight": 4}')
    check("手动来源被标记", resp.source == "manual" and resp.model.startswith("manual:"))
    check("手动模式带不可复现警告", "复现" in resp.request["warning"])

    llm = ManualLLMClient(tmp / "c" / "lp.txt", tmp / "c" / "la.json")
    try:
        llm.call("S", "U")
        check("LLM 手动客户端同样挂起", False)
    except PendingManualInput:
        check("LLM 手动客户端同样挂起", True)

    for alias in ("gemini-pro-latest", "gpt-preview", "some-model-stable"):
        try:
            ApiVLMClient(alias)
            check(f"滚动别名应被拒绝（{alias}）", False)
        except ValueError:
            check(f"滚动别名应被拒绝（{alias}）", True)
    c = ApiVLMClient("gemini-2.5-pro-2026-06-05", temperature=0.0, seed=7)
    check("固定版本号被接受", c.model == "gemini-2.5-pro-2026-06-05")
    meta = c.request_meta("s", "u", [])
    check("请求元数据记录版本/温度/种子",
          meta["model"] == c.model and meta["temperature"] == 0.0 and meta["seed"] == 7)
    try:
        c.call("s", "u")
        check("API 适配层未实现时明确报错", False)
    except NotImplementedError:
        check("API 适配层未实现时明确报错", True)


def test_gemini_client():
    """Gemini 适配层的离线部分。真正的调用只能在有 API Key 的 Linux 机上跑，
    但响应体解析可以用探针实测到的真实字段形状做回归。"""
    print("\n[6b] Gemini 适配层（离线）")
    cli = GeminiVLMClient("gemini-3.8-flash", temperature=0.0, seed=0, max_tokens=2048)
    check("视频采样率默认 12fps（1fps 下模型会把屈膝深浅答反）",
          cli.video_fps == GEMINI_VIDEO_FPS == 12.0, f"{cli.video_fps}")
    check("默认强制 JSON 输出", cli.json_mode)
    check("默认用完即删上传件（云端文件会过期，权威副本在本地）", cli.delete_uploads)
    check("LLM 侧不传素材故不删", GeminiLLMClient("gemini-3.8-flash").delete_uploads is False)

    # 以下字段形状来自探针 ⑨ 的真实返回体
    class _Det:
        def __init__(self, m, n): self.modality, self.token_count = m, n

    class _U:
        prompt_token_count = 4776
        candidates_token_count = 87
        thoughts_token_count = 742
        total_token_count = 5605
        prompt_tokens_details = [_Det("MediaModality.VIDEO", 4674),
                                 _Det("MediaModality.TEXT", 102)]

    class _Cand:
        finish_reason = "STOP"

    class _Http:
        headers = {"x-gemini-service-tier": "standard"}

    class _Resp:
        text = '{"weight": 3}'
        model_version = "gemini-3.8-flash"
        response_id = "VLbOatzULaTyvdIPqsLj-AY"
        candidates = [_Cand()]
        usage_metadata = _U()
        sdk_http_response = _Http()

        def model_dump(self, mode="json"):
            return {"model_version": self.model_version}

    body = cli._to_body(_Resp())
    check("提取 model 回显（用于检测静默降级）", body["model"] == "gemini-3.8-flash")
    check("提取 response_id", body["response_id"] == "VLbOatzULaTyvdIPqsLj-AY")
    check("提取 finish_reason", body["finish_reason"] == "STOP")
    check("提取 service_tier", body["service_tier"] == "standard")
    check("按模态拆分 token", body["usage"]["by_modality"] == {"VIDEO": 4674, "TEXT": 102},
          str(body["usage"]["by_modality"]))
    check("记录思考 token（实测远大于输出 token，计费必须算上）",
          body["usage"]["thought_tokens"] == 742)
    check("完整返回体原样落盘", body["raw"]["model_version"] == "gemini-3.8-flash")
    text, model, usage = cli._unpack(body)
    check("_unpack 三元组正确",
          text == '{"weight": 3}' and model == "gemini-3.8-flash"
          and usage["total_tokens"] == 5605)

    class _NoUsage:
        text = "x"
    check("缺字段时不崩", cli._to_body(_NoUsage())["usage"] == {})


# --------------------------------------------------------------------------- #
class _FakeRenderer:
    """假渲染器：只产出占位文件，验证循环的产物登记与同场对比拼图逻辑。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, motion_path, out_dir: Path, grid: GridSpec, tag: str):
        self.calls += 1
        imgs = out_dir / "images"
        imgs.mkdir(parents=True, exist_ok=True)
        grids = []
        for i, cam in enumerate(grid.cameras):
            p = imgs / f"{tag}_cam{i}.png"
            p.write_bytes(b"fake")
            grids.append(p)
        vid = out_dir / "video" / f"{tag}.mp4"
        vid.parent.mkdir(parents=True, exist_ok=True)
        vid.write_bytes(b"fake")
        return RenderOutput(grids=grids, video=vid)


def _fake_solver(params: GaitParams, out_dir: Path) -> SolveOutput:
    p = out_dir / "motion.pkl"
    p.write_bytes(b"fake")
    return SolveOutput(motion_path=p,
                       features={"knee_peak": 98.8 * params.knee_flex_scale},
                       status="converged")


def _scripted_client(answers):
    """按序返回预设答案的假客户端。"""
    box = {"i": 0}

    class C:
        name = "scripted"

        def call(self, system, user, media=()):
            a = answers[min(box["i"], len(answers) - 1)]
            box["i"] += 1
            return RawResponse(text=a, body={}, model="scripted-2026-01-01",
                               request={"media": [str(p) for p in media]},
                               source="api")
    return C()


def test_media_spec():
    """素材规格必须自己决定提交哪份产物。

    这里曾有一个不会报错的坑：prompt 正文已改成"以下是 2 段视频"，
    而 outer_loop 仍硬编码提交 6 张宫格图——模型会按文字描述去解读图片，
    给出看似合理实则错位的评分，全程不抛任何异常。
    """
    print("\n[6c] 素材规格：prompt 正文与实际附件必须一致")

    class _R:
        grids = [Path("a_cam1.png"), Path("a_cam2.png"), Path("a_cam3.png")]
        video = Path("a_video.mp4")

    class _NoVideo:
        grids = [Path("a.png")]
        video = None

    v, g = VideoSpec(), GridSpec()
    check("VideoSpec 取视频", v.select(_R()) == [Path("a_video.mp4")])
    check("GridSpec 取宫格图", len(g.select(_R())) == 3)
    check("两者的 artifact_key 不同",
          v.artifact_key == "video" and g.artifact_key == "grids")
    check("每个动作提交的份数",
          v.items_per_motion == 1 and g.items_per_motion == 3)
    try:
        v.select(_NoVideo())
        check("VideoSpec 遇到无视频应报错而非静默降级", False)
    except ValueError as e:
        check("VideoSpec 遇到无视频应报错而非静默降级", "视频" in str(e))

    p_v = render_vlm_prompt(3, 30.0, None, v)
    p_g = render_vlm_prompt(3, 30.0, None, g)
    check("视频 prompt 声明 2 段", "2 段视频" in p_v)
    check("宫格 prompt 声明 6 张", "6 张宫格图" in p_g)
    check("视频 prompt 说明时长差异是真实差异而非渲染错误",
          "不是渲染错误" in p_v)


def test_loop_end_to_end(tmp: Path):
    print("\n[7] 端到端假循环")
    run = tmp / "run_loop"
    if run.exists():
        shutil.rmtree(run)

    vlm_answers = [
        json.dumps({"weight": 2, "time": 3, "space": 3, "flow": 3, "naturalness": 9,
                    "comments": {"weight": "空载"}}),          # round 0 (A_ref)
        json.dumps({"weight": 4, "time": 3, "space": 3, "flow": 3, "naturalness": 8,
                    "comments": {"weight": "明显更沉"}}),      # round 1
        json.dumps({"weight": 1, "time": 2, "space": 2, "flow": 2, "naturalness": 5,
                    "comments": {"weight": "过度夸张"}}),      # round 2
    ]
    llm_answers = [
        json.dumps({"knee_flex_scale": 1.25, "trunk_lean_offset": 8.0, "cadence_scale": 0.9,
                    "double_support_offset": 0.05, "stride_length_scale": 0.85,
                    "elbow_flex_offset": 10.0, "tray_chest_distance_scale": 0.9,
                    "reasoning": "增加负重特征"}),
        json.dumps({"knee_flex_scale": 9.9, "trunk_lean_offset": 8.0, "cadence_scale": 0.9,
                    "double_support_offset": 0.05, "stride_length_scale": 0.85,
                    "elbow_flex_offset": 10.0, "tray_chest_distance_scale": 0.9,
                    "reasoning": "越界测试"}),
    ]
    vlm_box, llm_box = {"i": 0}, {"i": 0}

    def vlm_factory(idx, store, rep):
        c = _scripted_client([vlm_answers[min(vlm_box["i"], len(vlm_answers) - 1)]])
        vlm_box["i"] += 1
        return c

    def llm_factory(idx, store):
        c = _scripted_client([llm_answers[min(llm_box["i"], len(llm_answers) - 1)]])
        llm_box["i"] += 1
        return c

    renderer = _FakeRenderer()
    cfg = LoopConfig(run_dir=run, target_weight_kg=30.0,
                     termination=TerminationConfig(patience=1, score_target=0.99,
                                                   max_rounds=5,
                                                   ref_naturalness_floor=8),
                     reference_motion=tmp / "A_ref.pkl")
    loop = OuterLoop(cfg, vlm_factory, llm_factory, renderer, _fake_solver)
    loop.reference_features = {"knee_peak": 98.8}
    hist = loop.run()

    check("第0轮不调用求解器", hist.records[0].solver_status == "skipped(reference)")
    check("第0轮参数为恒等", hist.records[0].params.is_identity())
    check("第0轮成为初始最佳", hist.records[0].accepted)
    check("第0轮健康检查通过（未误报）", not hist.records[0].warnings,
          str(hist.records[0].warnings))

    check("第1轮被接受为新最佳", hist.records[1].accepted, hist.records[1].summary())
    check("LLM 参数被记录", hist.records[1].params.knee_flex_scale == 1.25)
    check("LLM reasoning 被记录", "增加负重" in hist.records[1].llm_reasoning)
    check("求解器特征被记录", abs(hist.records[1].features["knee_peak"] - 98.8 * 1.25) < 1e-6)

    check("越界参数被裁剪到上界", hist.records[2].params.knee_flex_scale == 2.0,
          str(hist.records[2].params.knee_flex_scale))
    check("低分轮次被回滚", hist.records[2].rolled_back)
    check("回滚后最佳仍为第1轮", hist.best.round_index == 1)
    check("停滞达标后终止", hist.should_stop() == StopReason.NO_IMPROVEMENT)

    # 同场对比：视频形态下第 N 轮应提交 2 段（历史最佳 + 候选）
    req = EvidenceStore(run).read_json(
        EvidenceStore(run).round_dir(2) / "vlm_response.json")["request"]
    check("第2轮提交 2 段视频（历史最佳 + 候选）", len(req["media"]) == 2,
          str(len(req["media"])))
    check("第一段来自历史最佳（第1轮）", "round001" in req["media"][0], req["media"][0])
    check("第二段来自本轮候选", "round002" in req["media"][-1], req["media"][-1])
    req0 = EvidenceStore(run).read_json(
        EvidenceStore(run).round_dir(0) / "vlm_response.json")["request"]
    check("第0轮只提交 1 段视频", len(req0["media"]) == 1, str(len(req0["media"])))
    check("提交的是视频而非宫格图", req0["media"][0].endswith(".mp4"), req0["media"][0])

    # 证据包完整性
    store = EvidenceStore(run)
    for f in ("params.json", "features.json", "record.json",
              "vlm_prompt.txt", "vlm_request.json", "vlm_response.json",
              "llm_prompt.txt", "llm_response.json", "vlm_score.json"):
        check(f"第1轮证据包含 {f}", (store.round_dir(1) / f).exists())
    check("run_meta.json 已落盘", (run / "run_meta.json").exists())
    check("history.json 已落盘", (run / "history.json").exists())
    check("留底视频已登记", "video" in hist.records[1].artifacts)
    check("宫格图仍照常登记（论文插图要用）", "grids" in hist.records[1].artifacts)
    check("run_meta 记录素材形态为视频",
          store.read_json(run / "run_meta.json")["media"]["kind"] == "video")

    # 断点续跑：从磁盘重建应得到同一状态
    loop2 = OuterLoop(cfg, vlm_factory, llm_factory, renderer, _fake_solver)
    check("重启后历史轮次数一致", len(loop2.history) == len(hist))
    check("重启后最佳轮次一致", loop2.history.best.round_index == hist.best.round_index)
    rebuilt = History.from_store(store, cfg.termination)
    check("从证据包目录可重建历史", rebuilt.best.round_index == hist.best.round_index)


# --------------------------------------------------------------------------- #
def test_noise_probe(tmp: Path):
    print("\n[8] 评分噪声摸底")
    run = tmp / "run_noise"
    if run.exists():
        shutil.rmtree(run)
    reps = [
        json.dumps({"weight": 2, "time": 3, "space": 3, "flow": 3, "naturalness": 9,
                    "comments": {"weight": "a"}}),
        json.dumps({"weight": 4, "time": 3, "space": 3, "flow": 3, "naturalness": 7,
                    "comments": {"weight": "b"}}),
        json.dumps({"weight": 3, "time": 3, "space": 3, "flow": 3, "naturalness": 8,
                    "comments": {"weight": "c"}}),
    ]
    box = {"i": 0}

    def vlm_factory(idx, store, rep):
        c = _scripted_client([reps[box["i"] % len(reps)]])
        box["i"] += 1
        return c

    def llm_factory(idx, store):
        return _scripted_client(["{}"])

    cfg = LoopConfig(run_dir=run, vlm_repeats=3,
                     termination=TerminationConfig(max_rounds=1),
                     reference_motion=tmp / "A_ref.pkl")
    loop = OuterLoop(cfg, vlm_factory, llm_factory, _FakeRenderer(), _fake_solver)
    loop.run()

    store = EvidenceStore(run)
    rd = store.round_dir(0)
    check("重复打分各自落盘", all((rd / f"vlm_rep{k}_response.json").exists() for k in range(3)))
    data = store.read_json(rd / "vlm_score_repeats.json")
    check("聚合取中位数（weight 2/4/3 → 3）", data["aggregated"]["weight"] == 3,
          str(data["aggregated"]["weight"]))
    check("聚合取中位数（naturalness 9/7/8 → 8）", data["aggregated"]["naturalness"] == 8)
    check("极差被记录（weight 极差 2）", data["spread"]["weight"] == 2.0,
          str(data["spread"]))
    check("无波动维度极差为 0", data["spread"]["time"] == 0.0)
    check("评语与中位数最接近的那次保持自洽",
          data["aggregated"]["comments"]["weight"] == "c",
          data["aggregated"]["comments"]["weight"])


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 70)
    print("motionedit 第4层（闭环）自测 — 不依赖 torch / opencv / 网络")
    print("=" * 70)
    tmp = Path(tempfile.mkdtemp(prefix="motionedit_loop_"))
    try:
        test_score()
        test_records(tmp)
        test_history(tmp)
        test_prompts()
        test_extract_json()
        test_clients(tmp)
        test_gemini_client()
        test_media_spec()
        test_loop_end_to_end(tmp)
        test_noise_probe(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"结果: {len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
