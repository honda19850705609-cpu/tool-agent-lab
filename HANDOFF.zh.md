# 🧠 项目交接记忆 · tool-agent-lab(中文)

> 换对话时把这份发给我,或直接说"继续 tool-agent-lab,看 HANDOFF / HANDOFF.zh;主线已出三发现 capstone,下一步=加难多步 agent 任务在顶部分高下"。

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

## 五、结果:一条三发现的能力主线(2026-06-13)
(多模态/VisDrone-VL 那条线已放弃,回归核心 LLM 主线。)把工具调用线从**单步 → OOD → 多步 agent** 推进,三个发现串成一个故事:**SFT 到底买到了什么。**

### 发现 1 —— DPO 接 SFT:提升微小,因为单步任务饱和了
建了完整 DPO 闭环(`build_prefs` on-policy/synthetic 偏好对;`dpo_lora` 用 SFT 模型当起点+冻结参考;`eval_toolcall --merge_adapter`)。结果(7B,n=300):exact_acc **0.883→0.890(+0.7,噪声内)**,全在参数上。教训:单步 xlam ≈0.88 **饱和**,测不动技术增益。*DPO 看着平的原因*:贪心解码吃掉了分布重塑;on-policy 负例只 542 个(SFT 在 82% prompt 上已全对);500 条验证集是地板。

### 发现 2 —— OOD 泛化:SFT 可迁移的是"可靠性",不是参数
在没训过的合成 8-tool zoo 上评。SFT *表面*掉(exact 0.933→0.820),但 `--show_errors` 显示 **20/20 是写法差异**(`Japanese→ja`、`3pm→15:00`),不是错值。拆解:SFT **强化且迁移了"选工具+格式可靠性"**(name/json→1.0),所谓"参数退步"只是约定漂移。结论:exact-match 部分测的是约定一致性而非能力 → 该用**执行式**评测。eval 加了 `arg_acc` 看这个。

### 发现 3(capstone)—— 多步 agent:可靠性连乘,微调 7B 打平 frontier
建了执行式多步评测(`tasks_multistep` + `eval_agent`):1–3 步任务锚在 `get_weather` 假数据上(必须真调工具链),按**任务办成没**打分。加 `agent/harmony.py` 让 **gpt-oss**(OpenAI,harmony 格式)走同一套评测。task_success(n=90):

| 模型 | 激活参数 | task_success | 备注 |
|---|---|---|---|
| Qwen2.5-7B base | 7B | 0.90 | `f_rise` 隐晦措辞下不主动调 get_weather |
| **Qwen2.5-7B + SFT** | **7B** | **1.00** | 毛病修好,还会从错误调用里自恢复 |
| Qwen2.5-14B | 14B | 0.989 | 差距主要是**尺寸** |
| gpt-oss-20b | 3.6B(MoE) | 1.00 | 用约 1/4 激活算力打平 14B = MoE 效率 |

易任务顶部全饱和(SFT-7B=14B=gpt-oss=1.0),于是加了 **HARD 集**(`generate(hard=True)`:3–5 步链 + `get_population` 干扰工具 + 比较逻辑),打破饱和、把模型**排出名次**(n=90):

| 模型 | 激活参数 | 易(总) | 难(总) | 难·4步 | 难·5步 |
|---|---|---|---|---|---|
| Qwen2.5-7B base | 7B | 0.90 | 0.644 | 0.40 | 0.53 |
| Qwen2.5-7B + SFT | 7B | 1.00 | 0.778 | 0.90 | 0.50 |
| Qwen2.5-14B | 14B | 0.99 | 0.856 | 0.80 | 0.77 |
| **gpt-oss-20b** | **3.6B(MoE)** | 1.00 | **0.967** | 1.00 | 0.90 |
| **Qwen3.6-35B-A3B** | **3B(MoE)** | 1.00 | **0.944** | 0.83 | **1.00** |

**三个干净、有机理的结论:**
1. **可靠性/深度墙真实**:dense 模型在最长链(5步)集体崩(base 0.53/SFT 0.50/14B 0.77),每步误差连乘(p^N)。
2. **"专精可抵规模"只成立一半**:SFT 把 7B 从 0.644→0.778(吃掉到 14B 差距的 ~38%)、在**中等深度(4步)追平大模型**(0.40→0.90),但**最深的 5 步暴露了微调补不上、得靠规模的上限**。易任务那个"SFT-7B=14B=1.0"是饱和假象。
3. **真 takeaway 是效率,而且是架构层的、不是单个模型**:**两个独立家族的 ~3B 激活 MoE**——gpt-oss-20b(3.6B,0.967)和当代 Qwen3.6-35B-A3B(3B,0.944)——都冲到顶部,并**双双越过 5 步深度墙**(0.90 / 满分 1.00),而所有 dense 模型(含 14B)都被拦在 ≤0.77,且只用约 1/4 per-token 算力。两家(OpenAI、阿里)复现 → **深度/可靠性墙是 dense 架构的限制;稀疏 MoE + reasoning 用 ~3B 激活就破了它。** 单位激活算力下最强的长链 agent 可靠性,是架构故事,不是"哪家厂商"的故事。

整条线闭环:DPO 边际(饱和)→ OOD 显示 SFT 可迁移的是*可靠性*非参数 → 多步显示可靠性*连乘* → 加难排出专精 vs 规模 vs (MoE)效率。**可选下一步**:剖析 SFT-7B 的 5 步失败(深度 5 到底崩在哪);或加"强制错误恢复 / 隐晦措辞"任务族;或就此收尾——故事已完整且有边界。

## 六、Colab/Drive 关键坑(切记)
- 运行时重置会清空 **/content + pip 包** → 重跑 Cell 0(重装 + 重挂 Drive);只有 **/content/drive 持久**。
- 下模型到 Drive:**别让 `local_dir` 直指 Drive**(双重缓存撑爆本地 ~235GB 盘 → Drive I/O 错 / 挂载失败)。正确姿势:**下到 /content → cp 到 Drive → 删本地 + 删 `~/.cache/huggingface`**;72B 这种超大才用 `--max_workers 2` 限速直写 Drive。盘满 / 挂载死 → 只能删除并重建运行时(Drive 文件不丢)。
- `pip uninstall -y torchao`(修 peft LoRA 报错;**加任何 adapter 前都要先卸**);`MsDataset.load` 有版本冲突 → prepare.py 改为直接读 json。
- **从 Drive(FUSE)加载大模型会卡死**(Llama-70B 134G 卡在 0/723)→ 先 `cp` 到本地 `/content` 再加载(顺序读稳)。
- **gpt-oss(MXFP4)加载**:要 `kernels>=0.12`+triton,但 `kernels` 版本坏了 → `pip uninstall -y kernels`+**重启** → transformers 自动反量化成 bf16(gpt-oss-20b ≈42G,96G 卡无压力)。分词器要 `tiktoken`。下载用 HF 的 `openai/gpt-oss-20b`(ModelScope 没这个 id,404)。
- **`pip install -U transformers` 后必须重启运行时**,否则内部 import 自相矛盾(如 `cannot import name GemmaQuantizationConfig`)。任何核心包升级同理。
- Llama-3.3-70B 的 ModelScope id = `LLM-Research/Llama-3.3-70B-Instruct`(免门控镜像)。70B dense 必须 4-bit 加载(`BitsAndBytesConfig`),bf16(~140G)装不下 96G。
- 这次环境是 **Colab 的 Blackwell 卡(RTX PRO 6000, 96G)**,sm_120;MXFP4/bnb 在 Blackwell 上偶有内核问题,bf16 反量化是稳妥退路。
