# 🧠 项目交接记忆 · tool-agent-lab(中文)

> 换对话时把这份发给我,或直接说"继续 tool-agent-lab,看 HANDOFF / HANDOFF.zh,下一步 SFT→DPO"。

## 一、背景与目标
- **上一个项目** `tiny-llm-quant-ablation`(已完结):从零造小 LLM 做量化消融,核心发现"**量化代价在能力边缘才出现**";并亲历"单卡从零训练 = 玩具"的规模墙。
- **当前项目** `tool-agent-lab`(GitHub: honda19850705609-cpu/tool-agent-lab):**微调开源基座 → 做一个能用的工具调用 agent**,研究方向 = **能力/对齐**(SFT / 数据配方 / DPO)。在 **Colab Pro+**(A100/H100/L4,高内存)上跑。

## 二、已完成
- 仓库结构:`agent/tools.py`(工具库)、`agent/runtime.py`(agent 循环,`Agent(base, adapter=...)`)、`data/prepare.py`(`--source synthetic|modelscope|hf`)、`data/synth.py`(零依赖合成数据)、`train/sft_lora.py`(TRL LoRA SFT,completion-only)、`eval/eval_toolcall.py`(json_valid/name_acc/exact_acc)、`COLAB.md`(Cell 0 = 重置后全套 setup)、`HANDOFF.md`(英文完整状态)。
- 跑通完整闭环:**真实数据(xlam,经 ModelScope 直连)→ LoRA 微调 → 受控评测**。

## 三、第一个研究结果(xlam 工具调用 exact-match,n=300)
| | 基座 | +LoRA | 增益 |
|---|---|---|---|
| Qwen2.5-1.5B | 0.703 | 0.810 | +0.107 |
| Qwen2.5-7B | 0.763 | 0.883 | +0.120 |

- SFT 提升**集中在"参数准确率"**(格式 ~1.0、选工具已高)。
- **有边界的结论**(单任务/单指标/同预算,未测泛化):
  - 同等条件 **7B>1.5B**(规模有用);**不能说"1.5B 优于 7B"**,也不能说"7B 整体更强"。
  - SFT 两个尺度都 ~+12 点,**7B 受益还略多**(相对错误↓51% vs 36%)——"小模型增益更大"的猜想**被推翻**。
  - **微调后 1.5B(0.810)> 裸 7B(0.763)= 专精可抵 ~5× 规模**(但这是"微调小 vs 没微调大"的不公平对比)。
  - 效率:微调 1.5B ≈ 微调 7B 的 92%,成本约 1/5。
  - Agent 可靠性 ≈ 每步准确率 p 的 N 次方 → 多步任务连乘衰减,需 p→~99% + 重试机制才从"demo"到"可托付"。

## 四、Drive 上的资产 `/MyDrive/Model/tool-agent-lab/`
- 模型:1.5B/3B/7B/Coder-7B/14B/32B/72B + **Qwen2.5-VL-7B-Instruct**(已下载,待用)
- 数据:`sft_train.jsonl`(20k)、`sft_val.jsonl`(500);adapter:`sft-qwen7b`、`sft-qwen1.5b`
- SFT 数据源:`AI-ModelScope/xlam-function-calling-60k`(`--source modelscope`,国内直连不门控)

## 五、下一步(进行中):SFT 之上加 DPO · 对齐梯子
**多模态/VisDrone-VL 那条线已放弃(2026-06-13)**,回归核心 LLM 主线。主线 = SFT → **DPO** →(数据配方 / OOD 泛化),继续精进工具调用模型的能力/对齐。

SFT 的增益集中在**参数准确率**且有天花板;DPO 训模型在"金标调用 vs 似是而非的错误调用"里选金标。代码已就位(都在 `main`):
- `data/build_prefs.py` —— 产 {prompt, chosen, rejected}。`--mode sampled`(**推荐**,on-policy:从 SFT 模型采样,留它**自己的错误**当 rejected,优先"工具名对/参数错"的硬负例)或 `--mode synthetic`(零依赖兜底:扰动金标参数)。**负例专打参数**。
- `train/dpo_lora.py` —— TRL DPOTrainer;把 SFT 合进权重,于是 **SFT 模型既是 DPO 起点又是冻结参考**(ref_model=None + peft_config),DPO 学一个新 LoRA。
- `eval/eval_toolcall.py` —— 加了 `--merge_adapter`(先把 SFT 烤进权重),于是 base / SFT / SFT+DPO 在同一验证集上同口径对比。`COLAB.md` 第 6–9 步跑完整闭环。
- 本地用 stub 烟测过 synthetic 负例逻辑(硬负例:单位互换、丢参、数值偏移;全部可解析、只在参数上不同)。`sampled`/DPO 需在 Colab 跑(GPU+trl)。
- **研究问题**:DPO 能否把 exact_acc 顶过 SFT?增益是不是落在**参数准确率**?注意过优化(LR 太高 / `--beta` 太低会把 json_valid 拉崩)。
- **下次开跑**:建偏好对(`--mode sampled`,1.5B 和 7B)→ DPO → 出三段式对比表。

## 六、Colab/Drive 关键坑(切记)
- 运行时重置会清空 **/content + pip 包** → 重跑 Cell 0(重装 + 重挂 Drive);只有 **/content/drive 持久**。
- 下模型到 Drive:**别让 `local_dir` 直指 Drive**(双重缓存撑爆本地 ~235GB 盘 → Drive I/O 错 / 挂载失败)。正确姿势:**下到 /content → cp 到 Drive → 删本地 + 删 `~/.cache/huggingface`**;72B 这种超大才用 `--max_workers 2` 限速直写 Drive。盘满 / 挂载死 → 只能删除并重建运行时(Drive 文件不丢)。
- `pip uninstall -y torchao`(修 peft LoRA 报错);`MsDataset.load` 有版本冲突 → prepare.py 改为直接读 json。
