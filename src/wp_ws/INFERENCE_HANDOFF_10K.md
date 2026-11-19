# 翻转枕头 ACT 10K：模型传输与真机推理交接

更新日期：2026-11-19  
交接对象：接收端推理开发者、机器人操作者  
交接状态：**离线候选模型，尚未完成真机部署验收**。

## 0. 本次交接的结论与范围

目标：将训练完成的 `act_mix8_b16_10k_v1/best.ckpt` 交到机器人侧笔记本，先完成接收校验和推理适配核验，再由操作者决定是否开展受控真机试验。

- 已确认：`best.ckpt` 的 step 是 **10000**；本仓库原生 `ACTPolicy` 支持其模型权重；固定验证集离线评估完成。
- 未确认：笔记本仓库版本、GPU/BF16 支持、真实机器人型号与驱动、相机映射、在线状态/动作转换、控制时序及安全停止行为。
- checkpoint 明确保存 `live_deployment_ready=false`，训练清单写明 **No deployment adapter is implemented**。不能将“权重可以加载”理解成“已有可直接启动的真机命令”。
- 本文的 Bash/Python 代码块已做语法检查；第 3.2 节静态验收已在训练机实跑通过（全部权重键匹配）。尚未执行网络传输或接收端验收。
- 本次只交接文件和流程；**不修改硬件、采集、动作执行代码，不启动或远程连接机器人**。下方传输及检查命令由用户执行。
- 不覆盖笔记本现有源码，不修改原数据、训练日志、checkpoint 或标定文件；不要为了通过检查而改写 checkpoint 的状态标志或代码指纹。

### 机器与目录

| 项目 | 训练机（发送端） | 笔记本（接收端） |
|---|---|---|
| SSH | 本机 | `ubuntu@192.168.20.118` |
| 项目 | `/home/ubuntu/Desktop/code/VLA/flash-aloha` | `/home/ubuntu/桌面/code/flash-aloha` |
| Conda | `/home/ubuntu/anaconda3`，环境 `flash-aloha-act` | `/home/ubuntu/miniconda3`，环境 `flash-aloha` |

所有模型、数据和日志根目录通过各机器自己的 `src/config/paths.py` 获取。已有 `FLASH_ALOHA_CKPT_DIR`、`FLASH_ALOHA_DATA_DIR`、`FLASH_ALOHA_LOG_DIR` 设置时，以其解析结果为准，不另写一套路径逻辑。

## 1. 要交接的模型

相对 `CKPT_DIR` 的位置：

```text
wp_ws/flip_the_pillow/act_mix8_b16_10k_v1/
├── best.ckpt                 # 本次使用；step 10000
├── run_config.json           # 训练配置、图像处理约定、实现指纹、来源信息
├── normalization_stats.json  # 仅训练集计算的标准化统计量
└── INFERENCE_HANDOFF_10K.md   # 本文；传输时额外复制，不是模型加载输入
```

**无需传 `last.ckpt`**：它约 749 MiB，包含训练恢复相关内容。`best.ckpt` 约 107.2 MiB，已含本次推理需要的模型参数。三份模型文件与本文合计约 108 MiB。最小交接不包含原始数据或离线评估图片。

### 固定身份（发送端已核验）

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `best.ckpt` | 112413771 | `42d67a27b38975327d614e3795ce8c912517579d2339bcfa169bd1577f0bbaa6` |
| `run_config.json` | 91840 | `03785aac98c169b42fdf87b0bf208f1aee4fbaa5c712dafd12af1a6899c65cbc` |
| `normalization_stats.json` | 5686 | `f8501942cdd92f86b00182febf8fa36600a6f1567a75e1398ff17631c27d3477` |

- 格式：`wp_ws.act_baseline.v1`。
- 模型：`repository_native_decoder_only_ACT_not_VAE_ACT`；不是含 KL 项的 VAE-ACT。
- 三相机独立 ResNet18，hidden dim 512，8 heads，7 decoder layers，dropout 0.1。
- `state_dim=14`、`action_dim=14`、`num_queries=50`，BF16；`training_time_rtc=false`。
- 保存的 camera 顺序：`cam_high`、`cam_left_wrist`、`cam_right_wrist`。
- 外置两份 JSON 与 checkpoint 内嵌的配置/统计量一致；外置清单中的发送端绝对路径是来源记录，不应被当作接收端可执行路径。

## 2. 传输命令：在训练机执行

下面只创建接收模型目录并传文件，**不会启动推理、驱动或回零**。SSH 首次连接时应核对主机身份。

```bash
(
  set -eo pipefail
  set +u  # Conda/ROS 激活脚本可能读取尚未定义的变量
  source /home/ubuntu/anaconda3/etc/profile.d/conda.sh >&2
  conda activate flash-aloha-act >&2
  set -u  # 环境激活完成后再开启未定义变量检查
  cd /home/ubuntu/Desktop/code/VLA/flash-aloha

  SRC=$(python -c 'from src.config.paths import CKPT_DIR; print(CKPT_DIR / "wp_ws/flip_the_pillow/act_mix8_b16_10k_v1")')
  for file in best.ckpt run_config.json normalization_stats.json; do
    test -s "$SRC/$file"
  done
  test -s src/wp_ws/INFERENCE_HANDOFF_10K.md

  DST=$(ssh ubuntu@192.168.20.118 'bash -s' <<'REMOTE'
set -eo pipefail
set +u  # 兼容 ROS activate.d 中未定义的 CONDA_BUILD 等变量
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh >&2
conda activate flash-aloha >&2
set -u
cd /home/ubuntu/桌面/code/flash-aloha
python - <<'PY'
from src.config.paths import CKPT_DIR
p = CKPT_DIR / 'wp_ws/flip_the_pillow/act_mix8_b16_10k_v1'
p.parent.mkdir(parents=True, exist_ok=True)
p.mkdir(exist_ok=False)  # 拒绝覆盖已有模型；已有同名目录则停止
print(p)
PY
REMOTE
  )

  printf '接收目录：%s\n' "$DST"
  scp "$SRC/best.ckpt" "$SRC/run_config.json" "$SRC/normalization_stats.json" \
    src/wp_ws/INFERENCE_HANDOFF_10K.md "ubuntu@192.168.20.118:$DST/"
)
```

**Conda/ROS 激活兼容性（2026-11-19 修正）：** 激活环境前显式 `set +u`，激活成功后才 `set -u`。这样兼容激活脚本读取未定义的 `CONDA_BUILD` 等变量，不需要修改 ROS 脚本、伪造构建变量或重装环境。`set -e` / `pipefail` 仍保留，激活本身失败就停止。激活提示转到 stderr，避免被 `DST=$(...)` 当作接收路径。这里仅在子 shell 内调整 shell 选项，不改变当前交互终端的设置。

若上一次报 `CONDA_BUILD: 未绑定的变量`，执行在远端激活阶段已停止，尚未执行该命令里的目录创建和 `scp`。直接完整重跑本节修正版；不要只补粘贴旧命令尾部。`overwriting variable ['PYTHONPATH']` 与这次未定义变量错误不是同一个问题；本命令不手动覆盖或清空 `PYTHONPATH`，后续以静态验收结果核实导入环境。

默认接收位置为：

```text
/home/ubuntu/桌面/code/flash-aloha/ckpts/wp_ws/flip_the_pillow/act_mix8_b16_10k_v1/
```

若命令报目录已经存在，不要删除原目录或简单改成 `exist_ok=True`：先确认它是既有模型还是本次中断留下的目录。`scp` 成功不等于模型验收通过，仍需执行下一节。整个命令放在子 shell 中，失败会停止本段，不会关闭当前交互终端。

## 3. 接收端静态验收：不连接硬件

### 3.1 环境检查

在**笔记本**执行：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate flash-aloha
cd /home/ubuntu/桌面/code/flash-aloha

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import sys
import torch
import torchvision
print('Python:', sys.version)
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16 supported:', torch.cuda.is_bf16_supported())
PY
```

训练环境供对照：Python 3.11.14、torch 2.7.1+cu128、torchvision 0.22.1+cu128、numpy 2.2.6、Pillow 11.3.0、opencv-python-headless 4.11.0.86。环境版本不一致不直接证明模型不可用，但必须做计算一致性验证；**不要盲目升级/重装机器人工作环境**。没有 CUDA/BF16 支持时仍可做下述 CPU 静态加载，不能据此承诺实时推理，也不要擅自改模型 precision。

### 3.2 哈希、清单和严格权重加载

仍在笔记本项目根目录执行。该代码只读文件、在 CPU 构建模型并加载参数，**不做在线推理，不导入机器人入口，不发送动作，不下载 ImageNet 权重**。

```bash
python - <<'PY'
import hashlib
import json
import math
import sys
from src.config.paths import CKPT_DIR, REPO_ROOT

root = CKPT_DIR / 'wp_ws/flip_the_pillow/act_mix8_b16_10k_v1'
expected = {
    'best.ckpt': '42d67a27b38975327d614e3795ce8c912517579d2339bcfa169bd1577f0bbaa6',
    'run_config.json': '03785aac98c169b42fdf87b0bf208f1aee4fbaa5c712dafd12af1a6899c65cbc',
    'normalization_stats.json': 'f8501942cdd92f86b00182febf8fa36600a6f1567a75e1398ff17631c27d3477',
}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

for name, digest in expected.items():
    require(sha256(root / name) == digest, f'文件哈希不符：{name}')
    print('SHA256 OK:', name)

config = json.loads((root / 'run_config.json').read_text())
stats = json.loads((root / 'normalization_stats.json').read_text())
# 检查最小计算依赖；不自动覆盖接收端源码，不伪造版本一致。
for key, relative in {
    'native_act': 'src/policy/act.py',
    'shared_images': 'src/contracts/standard_image_processing.py',
}.items():
    require(sha256(REPO_ROOT / relative) == config['implementation_sha256'][key],
            f'接收端实现与训练版本不同，请先审阅差异：{relative}')
    print('IMPLEMENTATION OK:', relative)

import torch
sys.path.insert(0, str(REPO_ROOT / 'src'))
from policy.act import ACTPolicy

torch.set_num_threads(2)
checkpoint = torch.load(root / 'best.ckpt', map_location='cpu', weights_only=True)
require(checkpoint['format'] == 'wp_ws.act_baseline.v1', 'checkpoint 格式错误')
require(checkpoint['step'] == 10000, '不是本次 10K checkpoint')
require(checkpoint['run_config'] == config, '内嵌与外置 run_config 不一致')
require(checkpoint['normalization_stats'] == stats, '内嵌与外置统计量不一致')
require(checkpoint['policy_config'] == config['policy_config'], 'policy_config 不一致')
require(stats['scope'] == 'train_episode_indices_only', '统计量来源错误')
require(stats['gripper_labels'] == 'preserve_raw', '夹爪标签约定错误')
for group in ('state', 'action'):
    require(len(stats[group]['mean']) == len(stats[group]['std']) == 14,
            f'{group} 标准化维度错误')
    require(all(math.isfinite(x) for x in stats[group]['mean']), f'{group} mean 非有限')
    require(all(math.isfinite(x) and x > 0 for x in stats[group]['std']),
            f'{group} std 无效')

policy = ACTPolicy(checkpoint['policy_config'])
result = policy.deserialize(checkpoint)  # 原生入口内部对 policy.model 使用 strict=True
policy.eval()
print('STRICT_LOAD_OK:', result)
print('step:', checkpoint['step'])
print('validation L1:', checkpoint['validation']['normalized_l1'])
print('live_deployment_ready:', checkpoint['live_deployment_ready'])
print('STATIC_CHECK_PASSED; 仅文件/CPU权重加载通过，未执行机器人动作。')
PY
```

期望：三个 `SHA256 OK`、两个 `IMPLEMENTATION OK`、`STRICT_LOAD_OK`、step 10000、`live_deployment_ready: False`。

这里的 `False` 是原始候选模型的正常状态，**不要手动改成 True**。若代码指纹不同，先 review 接收端差异；静态脚本拒绝继续不等于权重损坏。不得用 `strict=False`、删除权重键、关闭检查等方式伪装兼容。

## 4. 必须保持的训练/推理语义

### 4.1 图像

| 项目 | 本模型约定 |
|---|---|
| 训练相机顺序 | `cam_high` → `cam_left_wrist` → `cam_right_wrist` |
| 数据集列 | `observation.images.` 加上述名称 |
| 录制图像 | 原生 480×640×3，RGB |
| 共享预处理 | `src/contracts/standard_image_processing.py`，`Standard ACT/Flow` |
| 保存的配置 | `policy_config` 中 240×320；`aloha=false`、`grayscale=false` |
| 变换 | RGB、共享实现的插值参数 3、无 crop、无 letterbox、无训练增强 |
| 输入张量 | `[B, 3, 3, 240, 320]`，float32，uint8 / 255 |
| ImageNet 标准化 | 只在原生 `ACTPolicy._inputs` 内做一次 |

必须根据物理相机的序列号/位置显式建立映射，不能按字典顺序拼图，也不能把 `video_overhead` 等示例名字直接当作正确映射。核实接收端缓存实际为 RGB 还是 BGR，交由共享处理器按声明转换；不要在两端各写一份 resize/color/normalize。

`aloha=false` 在这里是**训练保存的图像处理开关**，不是机器人型号判断。不能因为机器人叫 ALOHA 就将它改成 True。图像尺寸相同也不代表颜色、视角、处理或采样时序一致。

### 4.2 状态、动作和标准化

```text
索引 0..5：左臂 6 个关节       索引 6：左夹爪
索引 7..12：右臂 6 个关节     索引 13：右夹爪
```

- 本数据语义：12 个关节是关节角/绝对关节目标，单位 rad；不是增量、笛卡尔坐标或速度。接收端仍需按原录制代码核实具体关节顺序和零点。
- 夹爪 state/action 都是原录制系统的归一化值，但两者来源和公式不同。**不能把全部 14 维当作弧度，也不能用 action 的公式构造 state。**
- `base_action` 不在模型输入/输出中，不控制底盘。
- 训练保留夹爪原值；action 左侧最高约 1.14531、右侧约 1.12546，出现大于 1 是已知情况，不是传输损坏。
- 使用 `normalization_stats.json` 的 `state.mean/std`、`action.mean/std`；它们只来自 91 个训练 episode、46111 行。`std` 已应用 0.01 下限，直接使用存储值，不重新估算。

```python
state_z = (state_raw - state_mean) / state_std
# policy(state_z, shared_processed_images) -> predicted_action_z
predicted_action_raw = predicted_action_z * action_std + action_mean
```

不得重复标准化、用验证/实时数据重算统计量，或将标准化输出直接发给电机。

**以下是用户提供的旧录制/回放脚本语义，不是本次硬件实测标定值：**

```python
# state：从臂单侧手指线性位置，单位 m
state_gripper = (single_finger_position_m - 0.01844) / (0.05800 - 0.01844)
# action：主臂夹爪电机角度，单位 rad
raw_action_gripper = (master_motor_angle_rad + 1.65) / 0.85
# 历史回放的从臂电机目标换算
historical_puppet_target_rad = raw_action_gripper * 1.491
```

先核实笔记本实际机器人、原录制/回放版本及已有驱动的转换和保护规则。**不要把最后一行直接复制成一个新驱动**。回放正常只证明已回放过的轨迹在当时条件下可执行，不证明任意模型预测安全。不要裁剪训练标签、绕过限位，或在本交接任务里更改夹爪转换来“适配”输出。

### 4.3 时间和 action chunk

- 数据声明 50 Hz，输出 chunk 长度 50。仅在声明频率成立时，50 个采样间隔名义上约 1 秒；这不是实测相机曝光同步或控制时序证明。
- 模型训练 `training_time_rtc=false`。不能未经验证启用 training-time RTC。
- 不能为了让现成示例启动而改成 100 Hz，或随意改 chunk、执行前缀长度、blending、mask、动作速率。必须先核实接收端既有执行策略能否保持录制数据对应的语义。
- 本次旧数据缺原始相机 frame ID/曝光时序 provenance；不能补造时间戳、重新配对或追认它已经满足标准采集契约。

## 5. 为什么不能直接运行仓库 `bimanual_act` 示例

以下是**发送端当前源码**的核查结果；笔记本须独立检查自身版本：

| 项目 | 仓库现成示例/入口 | 本次候选模型 |
|---|---|---|
| 任务配置 | `src/config/tasks/bimanual.yaml`：ARX L5，左右 can0/can1，100 Hz | 用户历史数据来自另一套 ALOHA 录制脚本；实际机器待核实，数据声明 50 Hz |
| 相机 | `video_overhead`、`video_left`、`video_right` | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| 部署配置 | `src/config/inference/bimanual_act.yaml` 指向 `all_configs.json`，示例 policy_step 100000 | `run_config.json` 是训练清单，step 10000，不是该入口的原生部署配置 |
| 统计量 | `inference_direct.load_policy_and_stats` 使用 `dataset_stats.pkl` 的平铺 qpos/action mean/std | JSON 中 `state`/`action` 的嵌套统计量 |
| 机器人合同 | 入口包含实际机器人合同检查、初始化/回零、采集和执行 | 尚未验证本数据/硬件的部署适配 |

`ACTPolicy.deserialize` 本身能识别 `model_state_dict`，并严格加载到内部 `policy.model`。权重由 `policy.serialize()` 保存，不能直接对外层 `policy.load_state_dict()` 加载后靠改键消除报错；**不是简单因为 checkpoint 有一层包装就无法加载**。真正缺的是已验证的部署配置、标准化接入、相机/状态/动作语义及硬件执行合同。

不要将 JSON 改名成 `all_configs.json`，不要只生成一个 pickle 就宣称适配完成，也不要直接用默认 `scripts/infer.py direct` 去试。入口可能在第一次预测前就初始化/回零。`live_deployment_ready=false` 是来源元数据，**不能假定所有入口都会自动阻止误用**。

若既有硬件/执行语义与模型不符，本轮停在边界，列出具体差异并另行确认范围；不能通过新 wrapper 或复制旧驱动绕过项目保护规则。

## 6. 接手开发者的工作顺序与验收门槛

### 阶段 A：文件与纯计算检查

1. 执行第 3 节，保存控制台输出；检查项目版本/工作区差异。
2. 确认模型、标准化、图像处理严格采用 checkpoint 约定；不下载/重新初始化预训练 backbone。
3. 只建立纯模型计算链：原始 state + 已确定的一组图像 → 标准处理 → 标准化 → `eval()`/inference mode → 输出 `[1, 50, 14]` → 反标准化。
4. **本阶段不调用任何 robot init、enable、home、reset、send-action、replay 入口。**

### 阶段 B：同一录制输入的离线一致性

最小四文件包不含验证输入，本阶段仍需另行提供一组或多组**原有录制样本**及参考输出。由开发者从源数据按原 row 关系提取；不能合成相机/状态时间关系。

- 同一观测在训练端和接收端比较共享处理后的图像张量、state_z、预测的标准化/反标准化动作。
- 样例至少覆盖初始、抓取、翻转、释放；不能只选静止片段。
- 图像前处理要求一致；模型数值差异要结合 BF16/设备/算子记录并制定容差，不能仅凭“shape 正确”验收。
- 完整验证集参考 L1 是 `0.39750448`。原离线评估的复现容差是 `0.0005 + 0.005 × 原 L1`，**只适用于同一完整指标复现，不是机器人安全阈值，也不是随便一帧输出的容差**。
- 如果录制样本在两端不一致，停止，不进入在线采集。

### 阶段 C：在线只观测、不发动作的 shadow 验证

由操作者执行**已经审阅、确实不触发动作的入口**；本次没有实现或提供该启动命令。

- 不能用会自动回零/使能/初始化运动的原生 runner 冒充 shadow 模式。
- 新/定制 live 入口必须调用 `src/contracts/standard_capture.py:capture_observation`：每臂一次 SDK 状态结果获得 qpos/qvel，随后取共享并行相机缓存快照，推理使用这次 observation 的图像。
- 不另读图像/本体填另一半，不选最近/更新帧，不由 FPS 构造时间戳。缺帧、过期、过大偏斜时 fail closed。
- 所有携带 live 数据的进程使用 `src/contracts/capture_gc.py`；stream 前暂停自动 GC，`collect_idle()` 仅在确认停止的 idle 阶段调用。
- 若保存现场数据，使用 `DistributedRecorder`、`RawEpisodeBuffer` 或标准 recording adapter；精确保留 `recording_fields()`，camera-server 编码；所有退出路径 drain/close。原始视频缺引用帧即导出失败，不修复配对。
- 记录相机物理映射、时间有效性、state/action 单位、预测有限性、幅度、相对当前状态的跳变和模型推理时延。此阶段不能据预测发送目标。

相关改动需通过硬件无关测试，并补实际适配的 parity 测试：

```bash
python -m pytest scripts/tests/test_standard_capture.py scripts/tests/test_standard_image_processing.py
```

测试通过不等于真实硬件验收通过；本次仅编写交接文档，没有声称已完成在线适配或跑过笔记本测试。

### 阶段 D：操作者受控试验

只有前述检查和现有硬件安全机制验收通过，才由操作者发起动作。**本交接不提供未经验证的 live 启动命令，也不代为启动。**

接手者先书面确认：

- [ ] 真实机器人型号、SDK、关节顺序/正方向/零点与录制语义对应。
- [ ] 当前夹爪 state/action 转换和硬件实测限位已核实；已有保护策略能处理本模型输出。
- [ ] 相机位置、安装、视角、序列号、色彩和模型输入映射已核实。
- [ ] 既有采集与执行入口保持标准契约；控制频率、chunk/RTC/blending 等没有被暗改。
- [ ] 起始位姿、工作空间、停止/失联/异常处理已由现场负责人验收；无自动误回零。
- [ ] 有可用急停，运动区域清空，由操作者值守；先做可随时停止的受控短试验，不无人值守连续执行。
- [ ] 任何 NaN、错误相机、时序失效、明显突跳、限位/保护报警都停止本次试验并保留证据，不在运行中临时改映射或关保护。

动作入口的具体命令应在以上信息确认后由接手开发者填写；如果实现它需要更改受保护的硬件/采集/执行行为，先报告边界，不在此交接范围里修改。

## 7. 现有评估证据与已知风险

固定划分：101 episodes / 51934 rows；训练 91 / 46111，验证 10 / 5823。验证 episode：

```text
[4, 8, 18, 19, 44, 53, 63, 89, 90, 92]
```

| 指标 | 数值 |
|---|---:|
| 完整验证 normalized L1 | 0.3975044836 |
| 训练集均值常数 L1 | 0.8427909054 |
| 相比常数基线误差降幅 | 52.83%，不是任务成功率 |
| 12 关节整体 MAE | 4.7499° |
| 两侧夹爪整体 MAE | 0.083438，原归一化单位 |
| 最差验证 episode | 4：L1 0.669561；8：L1 0.523529 |

- 验证结果可复现，说明离线评估链一致；不证明闭环稳定性、安全性或翻转成功率。
- 固定验证集参与过模型选择，不是独立测试集。图表是录制观测条件下的动作预测，不是模型闭环 rollout。
- 夹爪开合时刻和关键接触动作需要重点审阅，整体平均误差不能替代这项检查。
- 不应因目前 loss 已下降就把本次候选模型作为无人值守策略使用。

发送端原有证据（通过 `LOG_DIR` 定位）：

```text
wp_ws/flip_the_pillow/act_mix8_b16_10k_v1/
  report.md, COMPLETED.json, metrics.jsonl, split.json, continuity.json, fingerprints.json
wp_ws/flip_the_pillow/act_mix8_eval_v1/
  report.md, COMPLETED.json, metrics.json, examples.json, plots/, predictions/
```

其中 `plots/t0_episode_000004.png`、`plots/t0_episode_000008.png`、`plots/horizon_errors.png` 可辅助审阅。这些文件**不会由第 2 节最小命令自动传输**；如需复核，另附原报告/图片或按 `LOG_DIR` 配置路径传输，不改写证据。

## 8. 交接完成后请回传

1. 接收路径、三份文件哈希、step、严格加载结果；接收端软件/GPU版本及实现指纹差异。
2. 实际机器人和相机清单；状态/动作单位、夹爪转换及限位的核验结果。
3. 纯离线输入/输出 parity 结果与容差说明；shadow 阶段是否真正零动作及其时间/输出检查记录。
4. 拟使用的配置和入口命令，以及与标准采集/处理/执行合同的对应关系；未经核验不要填写“可部署”。
5. 受控试验的尝试次数、成功次数/定义、失败阶段、人工接管和报警记录。保存 checkpoint SHA、配置、帧/时间 provenance 与预测/实际执行值，不能仅回报“看起来正常”。

**本交接的下一项是接收端文件/纯计算验收；不是直接用示例配置启动真机。**
