# 翻转枕头：离线数据体检与固定划分

本目录只做**训练前的只读检查**，不启动机器人、回放、采集或训练。

- 对象：当前 `data/flip_the_pillow` 中的 **LeRobot v2.1、嵌入式 RGB 图片、14 维 state/action**。
- 原始 Parquet、图片、动作、状态、时间戳、任务元数据均不改写。
- 不裁剪夹爪、不换算动作单位、不用 state 替代 action、不移动标签、不重新配对图像、不重采样。
- 全量解码图片，不以抽样检查冒充全量通过。不依赖 CUDA、torch、ROS 或 LeRobot。
- 代码由此目录提供；**实际数据体检和测试由使用者手动运行**。

## 1. 环境

在 Flash ALOHA 项目根目录执行，`python` 应为你选定的 Python 环境（建议 Python 3.10+）：

```bash
cd /home/ubuntu/Desktop/code/VLA/flash-aloha

# 若当前环境还缺少依赖，再执行；不会安装训练框架或机器人驱动。
python -m pip install -r src/wp_ws/requirements.txt
```

如果在服务器上运行，将 `cd` 换成服务器上的项目目录。应保留项目的
`src/config/paths.py`，不要只复制一个脚本然后硬编码数据地址。

数据与日志目录统一来自 `src/config/paths.py`：

- 默认输入：`data/flip_the_pillow/`
- 默认输出：`logs/wp_ws/flip_the_pillow/audit_v1/`
- 需要在服务器上换根目录时，使用该模块已有的
  `FLASH_ALOHA_DATA_DIR` / `FLASH_ALOHA_LOG_DIR` 环境配置；数据根目录下仍放
  `flip_the_pillow/`。脚本不另设 `--data-dir` / `--output-dir` 覆盖中央配置。
- 输出目录不能位于源数据集内。

## 2. 先跑合成数据测试（可选但建议）

```bash
python -m unittest discover -s src/wp_ws/tests -v
```

测试只在临时目录创建很小的 Parquet 和图片，不访问真实演示，不连接硬件。
覆盖固定划分、数据只读、超参考范围标签、坏图、NaN、错误维度/索引、时间倒退、
暂停/时间间隔边界、源文件指纹和禁止覆盖旧报告等情况。

## 3. 正式体检命令

先确保这份数据已经停止采集/写入；不要一边追加 episode 一边体检。

```bash
python -m src.wp_ws.audit_dataset \
  --dataset flip_the_pillow \
  --expected-task flip_the_pillow \
  --seed 42 \
  --val-episodes 10 \
  --run-name audit_v1
```

命令会逐 episode 显示进度。默认完整解码每帧的每路图片，并计算所有声明的
Parquet 和相关元数据文件的 SHA-256。需要 CPU、内存和磁盘读取，不需要 GPU。
`--batch-rows 64` 是流式读取批大小，**不是训练 batch size**。

已有同名输出目录时拒绝覆盖。重跑可改用 `--run-name audit_v2`；同样的 episode IDs、
seed 和验证集数量将得到同样的 episode 划分。不要删除旧报告来掩盖数据版本变化。

### 固定划分规则

- 默认 seed=42，验证集 10 条。当前若 101 条均检查通过，就是 **91 条训练、10 条验证**。
- 采用版本化 `sha256-episode-rank-v1`：对每个 episode ID 计算
  `SHA256("sha256-episode-rank-v1:SEED:ID")`，按摘要字节排序，前 10 条作为验证集。
- 不依赖 Python/NumPy 的随机数实现，不受 `episodes.jsonl` 行顺序影响。
- 分区单位是**整条 episode**。训练/验证无重复、全集覆盖，不按帧随机拆分。
- **任何结构性错误都会阻止生成 split.json**，不悄悄丢弃失败的 episode 再重分。
- 原始 `meta/info.json` 的 `splits.train` 不改写。后续训练加载器必须**显式使用
  本次 split.json 的 ID 列表**；仅执行本工具不会让其它训练命令自动采用此划分。
- 这是 episode 级基线，不是按录制 session 或场景分组的泛化评测；工具不凭空推断缺失的
  session 标签。将来需要分组评测时，应另立明确版本，不能偷偷改动本划分。

## 4. 输出文件

```text
logs/wp_ws/flip_the_pillow/audit_v1/
├── report.md          # 首先阅读：摘要、夹爪范围、错误/警告
├── report.json        # 完整结构化结果，逐 episode 数值/图像/时间检查
├── fingerprints.json  # 源文件相对路径、大小、SHA-256
├── continuity.json    # 保存的连续行区间；没有修复或改写原始样本
├── split.json         # 仅在没有 error 时生成；固定训练/验证 episode IDs
└── COMPLETED.json     # 最后写入；缺失表示报告发布未完成
```

`COMPLETED.json` 存在**不等于体检通过**，仍需检查其中的 `status`、`split_ready`。
中断后的目录不自动复用；换一个 `--run-name` 重跑。

源文件指纹使用相对路径和完整内容摘要，因此搬到服务器换绝对路径不会改变指纹。
不同内容/元数据版本会产生不同指纹。保留报告、指纹和划分，将来与训练记录一起归档。
本轮只生成检查结果，**不计算训练标准化统计量**；报告中的全数据范围只用于诊断。
后续必须仅用 `train_episode_indices`，分别拟合 state/action 的标准化统计量。

## 5. 如何判读

退出码：

- `0`：检查完成且可生成划分；可能为 `PASS_WITH_WARNINGS`，仍需读警告。
- `2`：数据体检失败；报告尽量保留，但没有 `split.json`。
- `1`：环境依赖、参数、输入根目录或输出目录等运行问题。

### 阻止生成可用划分的错误

- 元数据不合法，episode 数量/总帧数不符，文件缺失、重复引用或未提交 Parquet。
- state/action 等已知数值字段的类型/维度不符、null/NaN/Inf。
- episode/frame/global index 不一致，未知 task_index。
- 图片缺少内嵌字节、无法完整解码，或 RGB 尺寸与声明不符。
- elapsed_time 不递增、segment 倒退或与 episode 元数据不符。
- 标称 `timestamp` 不符合该录制器定义的 `frame_index / fps`。
- 体检期间检测到源文件变化。

注意：最后一项时间等式是对已有录制字段的**一致性检查**，不是按 FPS 生成/替换相机
曝光时间。报告不恢复相机时间戳或帧 ID，不证明相机与本体同步，也不宣称符合
Flash ALOHA 原生采集契约。

### 不自动删数据的警告

- 夹爪 state/action 超过参考 `[0,1]`：记录两侧原始范围和数量，**不裁剪、不判为坏标签**。
- 源 task 为 `pick_cup` 而预期为 `flip_the_pillow`：如实记录，不改元数据，
  不假装已完成后续训练器的任务名映射。
- 保存的 elapsed_time 出现较大间隔：默认阈值 `3 / fps`，50 Hz 时是 60 ms。
  `continuity.json` 记录 episode、segment 和此类间隔形成的边界，训练 action chunk
  应留在区间内。阈值是诊断配置（`--gap-factor`），不是原始相机时序证明。
- 检测到非零 base_action：它是测得的底盘速度，不是已经确认过的底盘控制标签；
  使用仅双臂 baseline 前应审阅。

连续区间采用 **`[start_frame, stop_frame_exclusive)`**，不丢弃边界前后任何行。
`continuity.json.usable` 为 false 时不能拿它继续训练。

### 动作语义保持不变

| 字段 | 原始含义 |
|---|---|
| action 的 12 个机械臂维度 | 主臂绝对关节角，rad |
| action[6] / action[13] | `(主臂夹爪电机角 + 1.65) / 0.85`，无量纲，不裁剪 |
| state[6] / state[13] | `(从臂单侧手指位置 - 0.01844) / (0.05800 - 0.01844)`，无量纲 |
| 执行映射参考 | 从臂夹爪目标角 = 原始 action × 1.491；训练标签不做这个换算 |

本工具不读取或修改硬件寄存器，不验证机械硬限位，不执行裁剪/速度限制，不修改原有
回放/采集/控制代码。回放正常也不等于每条演示在语义上都成功；需要人工审阅的失败演示
不会被工具凭空识别或静默剔除。

## 6. 本步骤完成后

先查看 `report.md`。如有错误，把 `report.json` 的 `issues` 发回来；如果只有已知警告，
再准备显式消费 **split.json + continuity.json** 的离线 ACT 加载器、仅训练集标准化和
服务器短测。届时图像确定性处理必须复用 `src/contracts/standard_image_processing.py`；
本工具只检查原始图片，没有另建训练/推理预处理实现。

## 7. 当前决定：不改源任务名，训练侧配置翻转枕头

**不运行改名脚本、不改源元数据、不改旧报告。** 源 `task` 保留 `pick_cup`，
训练实验名用 `flip_the_pillow`，中文描述为“翻转枕头”。只要数据及体检文件没有变化，
继续使用 `audit_v1`；不需要因为这个训练别名重新体检。

数据配置：`src/wp_ws/flip_the_pillow_baseline.json`，由
`src/wp_ws/baseline_data_config.py` 解析。两个字段有意分开：

```json
{
  "task_name": "flip_the_pillow",
  "task_description": "翻转枕头",
  "source_task_name": "pick_cup",
  "dataset": "flip_the_pillow",
  "audit_run": "audit_v1"
}
```

以上只是完整配置的关键字段，不要用这个片段覆盖完整文件。`task_name` 是实验/
checkpoint 命名，不覆盖读取到的 `task_index` 或源标签，不作为额外的动作转换。
后续加载器须核对 `source_task_name`，并在训练产物中保留源标签、训练别名和源指纹。
当前的 `TASK_LABEL_MISMATCH` 警告如实保留，不删除、不伪装成已经改名。

配置预览（不需要 GPU/第三方依赖，不读取演示或体检内容，也不创建目录）：

```bash
python -m src.wp_ws.baseline_data_config
```

输出 `CONFIG_PREVIEW_ONLY` **不代表训练就绪或指纹验证通过**。本配置不是项目原生
训练 YAML，不能直接传给原生训练命令；兼容加载器与离线训练入口见第 8 节。
项目现有标准加载器会自行随机拆分，不能直接代替已固定的 `split.json`。

初始 baseline 的数据约定已固定为：

- 输入 `observation.state` 为原始 14 维；监督 `action` 为原始 14 维。
- `use_base_action=false`：不把测得的底盘速度拼入 state/action，不补零、不修改源列。
- 夹爪标签 `preserve_raw`：不裁剪、不换算、不用 state 替换 action。
- 显式读取 `audit_v1/split.json`，不按 `train_ratio` 重新划分；标准化只拟合训练集。
- 使用 `audit_v1/continuity.json`，后续 action chunk 不跨 episode/segment/时间间隔边界。
- 第 8 节加载器复用共享 `standard_image_processing` 的 Standard ACT/Flow profile，
  并在 checkpoint 保存相同的预处理配置；尚未实现针对这套数据/机器人的部署适配器，
  不据此声称端到端部署已经打通。
- 所有数据、日志和 checkpoint 根目录继续取自 `src/config/paths.py`；不在此配置写绝对路径。

服务器补充检查：`split_ready=True`、`continuity_usable=True`；底盘线速度范围约为
`[-0.001, 0] m/s`，角速度约为 `[-0.004, 0.002] rad/s`。这些极值很小，
但不能仅凭极值判断是测量噪声，也不能证明底盘完全静止或累计位移可忽略。
`BASE_MOTION_PRESENT` 继续保留；应结合录制过程/视频确认底盘没有参与完成任务。
不因这些数值擅自裁剪、删帧、重录或更改数据。

可由使用者手动运行配置合成测试：

```bash
python -m unittest src.wp_ws.tests.test_baseline_data_config -v
```

现在已实现上述离线加载与训练代码。**下一步由使用者运行第 8 节的合成测试、preflight，
再进行 CUDA 短训练**；不要改用项目原来的随机划分加载器。

## 8. 离线 ACT baseline：从预览走到真实数据短测

### 本次实现范围

| 文件 | 职责 |
|---|---|
| `verified_data.py` | 核验完成标记、源文件 SHA-256、split 完整/不重叠、真实连续区间 |
| `act_dataset.py` | 直接读取原始 Parquet；训练集统计；边界 padding/mask；共享图像处理 |
| `train_act.py` | `preflight` 与原生 ACT CUDA/bf16 短训练，固定验证、日志和 checkpoint |
| `tests/test_act_baseline.py` | 合成数据的指纹/划分/连续性/标签/图像/只读性测试 |

**不需要转存 HDF5/Zarr、不修改任务标签、不重新导出图片、不改原始夹爪值。**
这是仓库的 **decoder-only ACT**，不是另装 LeRobot 的 VAE-ACT。直接复用
`src/policy/act.py` 及其 `MasterWeightAdamW`：模型/梯度 bf16，FP32 主权重与 AdamW
状态；不另造优化器、不启用 RTC、compile、随机图像增强或新的图像 profile。

默认三路相机顺序为 `cam_high / cam_left_wrist / cam_right_wrist`，原始 RGB bytes
逐行解码，通过共享 Standard ACT/Flow profile 的 INTER_AREA resize 转为
`3×3×240×320`、float32 `[0,1]`；ImageNet 标准化仍在原生 ACT 内部完成。
不裁剪、无 letterbox、无灰度。`--height/--width` 是明确的训练配置，会写进 checkpoint，
不能在部署时任意改大小。原始 480×640 图像不被覆盖。

训练时每个 epoch 先 shuffle episode，再 shuffle 该 episode 的行，并在 episode 内
组 batch，避免随机跨文件导致反复读取大 Parquet。每行每 epoch 采样一次，不丢尾部 batch。
每个加载进程最多缓存一个 Arrow episode（保存的压缩图片字节，不是全量解码图像）；
实际 RAM 占用仍取决于单个 episode 大小、worker 数与预取，初次请用 `--workers 0`。
默认 chunk 为 50 个保存动作，50 Hz 下为 **1 秒名义时间**，不是相机曝光同步证明。
接近 episode/segment/elapsed-gap 边界时保留当前行，用零填充**标准化后的 padding**，
并在 loss 中完全 mask；不跨边界补动作，也不丢弃尾部短 chunk。

### 8.1 环境依赖（仅缺包时补装）

在服务器已安装的 `(flash-aloha)` Conda 环境内运行。先检查，不要无条件覆盖现有 CUDA 栈：

```bash
python - <<'PY'
import torch, torchvision, cv2, pyarrow
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('opencv:', cv2.__version__, 'pyarrow:', pyarrow.__version__)
print('cuda:', torch.cuda.is_available())
print('bf16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

离线 audit 的 requirements 不含训练依赖。需要新装时可使用
`python -m pip install -r src/wp_ws/requirements-act.txt`；其中 torch/torchvision 与项目
`pyproject.toml` 的固定版本一致，但 GPU 驱动、CUDA wheel 来源应按服务器情况选择，
不要为了这一步安装机器人驱动、ROS 或整个 LeRobot 依赖栈。原生 ACT 只接受 bf16；
不支持时会明确报错，不静默降到 fp16。preflight 本身不需要 torch/CUDA。

### 8.2 先跑合成测试，再核验真实数据（使用者执行）

```bash
cd /data1/wp_ws/flash-aloha
conda activate flash-aloha

# 仅 TemporaryDirectory 合成 Parquet；不接触真实演示、不训练。
python -m unittest discover -s src/wp_ws/tests -v

# 实际读取服务器数据，但不改写、不训练、不连接机器人。
python -m src.wp_ws.train_act preflight \
  --run-name act_preflight_v1 \
  --batch-size 4 --chunk-size 50 --height 240 --width 320
```

preflight 会：

1. 读取 `COMPLETED.json / report.json / fingerprints.json / split.json / continuity.json`。
2. **重新计算当前源文件 SHA-256**，与 `audit_v1` 比较；不重新全量解码 155802 张图片。
   有逐文件进度；大文件 hash 需要时间，不能用路径/文件存在代替内容验证。
3. 从保存的 elapsed_time 与 segment 验证区间，检查固定划分覆盖所有 episode、无重叠。
4. 仅遍历训练集 state/action 列，以 float64 合并总体矩计算 mean/std；std 下限 `0.01`
   用来稳定常量维度。**这不是裁剪动作**；state/action 各自统计，验证集完全不参与。
5. 解码训练/验证 CPU batch，检查形状/有限值，输出 `PREFLIGHT_PASSED`。

默认日志：`logs/wp_ws/flip_the_pillow/act_preflight_v1/`。其中包含：
`preflight.json`（batch 形状/原始夹爪反标准化范围）、`normalization_stats.json`、
`run_config.json`、原审计划分/连续性/指纹快照，最后写 `COMPLETED.json`。
预览 `CONFIG_PREVIEW_ONLY` 与这里的内容指纹核验是不同阶段。
**PREFLIGHT_PASSED 不代表 CUDA 已能训练，更不代表机器人动作安全或任务成功。**

### 8.3 100 步短训练：实测显存（使用者执行）

审阅原有警告后再运行。`--acknowledge-audit-warnings` 表示知悉警告，不删除或忽略
指纹/连续性错误，也不把旧采集数据伪装成符合原生时序契约。`BASE_MOTION_PRESENT`
仍需结合视频/录制过程确认；此训练不预测或控制底盘。

```bash
python -m src.wp_ws.train_act train \
  --run-name act_smoke_v1 \
  --steps 100 --batch-size 4 --chunk-size 50 \
  --height 240 --width 320 --workers 0 \
  --validate-every 50 --val-max-batches 10 \
  --acknowledge-audit-warnings
```

- 默认用 ImageNet `IMAGENET1K_V1` 初始化 ResNet18；torchvision 可能尝试联网下载，缓存
  由中央路径配置到 `CKPT_DIR/wp_ws/_torchvision_cache`。若网络不通，准备同一版本权重缓存，
  不会静默转为随机权重。仅做计算连通性测试时可**显式**选择 `--backbone-init scratch`，
  但它不是相同的 pretrained baseline，不要混淆实验结果。
- train 每次启动仍重新验证源 SHA-256、重新拟合训练统计：不盲信旧 preflight，不能在两次
  命令之间改数据或审计文件。固定训练/验证仍为原 91/10 episodes，不用 train_ratio。
- 默认 `val-max-batches=10` 只评估固定验证集上**均匀间隔的最多 40 个起始行**，覆盖整个
  验证序列范围，而不是随机抽样。样本索引写入 run_config；这是快速 smoke 指标，
  **不是 5823 行全量验证**。loss 为有效动作元素加权的 normalized L1，不是成功率。
- `metrics.jsonl` 与终端会输出 `peak_allocated_gib`、`peak_reserved_gib`，包括模型、
  FP32 master weights、优化器状态、反向传播和验证阶段。它们是本进程 PyTorch 峰值，
  不是整卡所有进程/驱动的总显存；应另外留余量，并结合 `nvidia-smi` 查看整卡占用。
- 当前没有服务器实测值，不把推测显存当作结果。如果 OOM，把 batch 降为 2，再使用新名字
  （如 `act_smoke_b2_v1`）；不要改原始数据或绕开 bf16 要求。

默认产物：

```text
logs/wp_ws/flip_the_pillow/act_smoke_v1/
  preflight.json / normalization_stats.json / run_config.json
  split.json / continuity.json / fingerprints.json
  metrics.jsonl / COMPLETED.json
ckpts/wp_ws/flip_the_pillow/act_smoke_v1/
  run_config.json / normalization_stats.json
  best.ckpt / last.ckpt
```

两个根目录都只来自 `src/config/paths.py`。已有 run 不覆盖；省略 `--run-name` 可自动
生成 UTC 时间名。失败/中断日志写 `FAILED.json`，不会写训练完成标记。

### 8.4 短测通过后才开正式 baseline

依据实测显存确定 batch，确认短测 loss 有限且无加载/指纹错误后，再使用**新 run**。
以下仅为起始方案，不表示 30000 步足够学会任务：

```bash
python -m src.wp_ws.train_act train \
  --run-name act_baseline_v1 \
  --steps 30000 --batch-size 4 --chunk-size 50 \
  --height 240 --width 320 --workers 0 \
  --validate-every 2000 --val-max-batches 0 --save-every 2000 \
  --acknowledge-audit-warnings
```

`--val-max-batches 0` 才是所有固定验证起始行，padding 元素仍不算 loss。全量验证也不
用验证数据拟合统计。`best.ckpt` 按该 run 的固定验证 normalized L1 选取；
`last.ckpt` 另保存优化器/部分 RNG 状态供后续开发使用，**当前尚未实现 resume 入口**，
不要以为中断后直接重复命令就能续训（已有目录会拒绝覆盖）。

这些 checkpoint 明确标记 `live_deployment_ready=false`，保存了原始标签语义、标准化、
图像配置、来源指纹、代码指纹和任务别名。**尚未实现面向本套数据/机器人的在线推理适配，
不能直接交给原有机器人推理入口启动动作。** 后续部署必须在同一共享图像 profile、
同一相机顺序/动作语义与安全边界内单独验证；本次不改动任何采集、回放、执行、限位或硬件代码。

## 9. W&B 曲线：独立读取指标，不重启训练

`wandb_metrics.py` 是旁路上传器：只读已有 `metrics.jsonl` 和白名单实验配置，
**不启动训练、不连接机器人、不需要 GPU，也不改写原训练目录**。训练已经开始或
结束都可以使用，不需要为加 W&B 中断当前训练（训练入口仍未实现断点续训）。

先把本目录的新文件同步到服务器，在当前训练所用的 Python/Conda 环境安装可选依赖：

```bash
python -m pip install --upgrade -r src/wp_ws/requirements-wandb.txt
python -c "import importlib.metadata as m; print('wandb', m.version('wandb'))"
python -m wandb login --relogin
```

可选依赖固定为 `wandb==0.22.3`：这是开始支持超过 40 字符 API Key 的版本。
若旧环境报 `API key must be 40 characters long, yours was 86`，先执行上面的
升级命令并确认版本为 `0.22.3`，再重新登录；**不要截断 Key，也不要修改 SDK
的长度校验**。这个报错不能证明 Key 本身有效，登录时仍需只粘贴完整 Key，
不要带引号、`export`、变量名或其他文字。前面的 Pydantic
`UnsupportedFieldAttributeWarning` 不是本次退出的原因，不必为此重装训练环境。

请在能读取原训练日志的机器上运行上传器；本机登录不会替远程服务器登录。
若训练在服务器，通常直接在服务器的对应 Conda 环境安装、登录并运行上传器，
无需停止或重启原训练，也不需要重装 PyTorch。

登录密钥只在你自己的终端输入，不写进脚本、训练配置或聊天。
`online` 会将白名单标量、实验配置和 W&B 自身运行日志发送到你的 W&B 服务，
使用前确认数据出域规则；默认 `offline` 不上传。

### 9.1 已在运行的正式训练：导入历史并跟随

在**另一个终端**、项目根目录运行；`run-name` 必须与已有训练目录一致：

```bash
python -m src.wp_ws.wandb_metrics \
  --run-name act_baseline_gpu1_v1 \
  --project flash-aloha \
  --mode online \
  --follow
```

需要指定 W&B 账号/团队时，加 `--entity 你的账号或团队`；不填则使用账号默认设置。
上传器打印 W&B run 页面，在项目 Workspace 中查看：

- `train/normalized_l1`：训练 batch 损失；
- `val/normalized_l1`：验证损失，summary 记录最小值；
- `train/gradient_norm_before_clip`：裁剪前梯度范数；
- `cuda/peak_allocated_gib`、`cuda/peak_reserved_gib`：原训练进程报告的 PyTorch 显存峰值。

这些曲线的横轴均为原训练的 `global_step`，不是上传器的消息计数；同一步的
train/validation 都会保留。上传器只转发已记录的点，不补造中间 loss；更新频率仍由
原训练的 `--log-every` / `--validate-every` 决定。需要叠加 train/val 时在同一个
Line Plot 中选择上述两条 loss，X 轴选 `global_step`；平滑只用于显示，不修改原值。
验证子集/完整验证模式写入 run config/tags，不把两种指标混为同一评价口径。

`--follow` 从文件开头导入历史，然后等待追加内容；并发写入的半行会等待写完再读。
训练完成/失败后自动结束跟随。训练若被强杀且没有终态标记，上传器可能继续等待，
此时可在**上传器终端**按 Ctrl+C；这只停止上传器，**不会停止原训练**。

### 9.2 已结束的 100 步短测：只导入现有曲线

```bash
python -m src.wp_ws.wandb_metrics \
  --run-name act_smoke_gpu1_v1 \
  --project flash-aloha \
  --mode online
```

没有 `--follow` 时只导入本次读取到的完整记录；若训练仍在运行，这只是当前快照，
不表示训练已结束。每次调用创建独立的 W&B run，不恢复/合并此前上传器；重复执行
会得到多个显示名相同但 ID 不同的 W&B run。

### 9.3 无外网及文件位置

无外网时将上例改为 `--mode offline`；离线记录不会实时出现在网页上。
以后可用 `wandb sync` 同步脚本输出位置中的 `wandb/offline-run-*` 目录。
上传器文件只写到：

```text
LOG_DIR/wp_ws/_wandb_uploads/<task>/<source-run>/upload-*/
```

`LOG_DIR` 仍来自 `src/config/paths.py`，无额外日志根目录选项；不受 `WANDB_DIR`
另行改址，SDK 内部缓存/暂存目录也限定在该上传目录下。脚本不调用 `watch()`、
`save()` 或 artifact 上传，不上传源图像、
state/action、标准化统计量或 checkpoint；代码/git/console 自动收集与上传器自身
系统监控均关闭。原训练的 `metrics.jsonl` 仍是指标来源，不依赖 W&B 能否联网。

合成日志测试（不联网、不需要安装 W&B、不训练）：

```bash
python -m unittest src.wp_ws.tests.test_wandb_metrics -v
```

## 10. 本机启动脚本：batch 16 短测和 10 万步 baseline

项目的脚本目录名是 `scripts`；入口为
`scripts/wp_ws/train_flip_the_pillow_act.sh`。先激活自己的 Conda 环境，再执行：

```bash
conda activate flash-aloha-act

# 只打印命令；不运行 Python、不下载权重、不访问 GPU、不创建训练目录。
bash scripts/wp_ws/train_flip_the_pillow_act.sh smoke --dry-run

# 先实测 batch 16：100 步，快速验证子集。
bash scripts/wp_ws/train_flip_the_pillow_act.sh smoke

# 短测通过后手动启动：100000 步，全量固定验证，每 5000 步验证/保存。
bash scripts/wp_ws/train_flip_the_pillow_act.sh train
```

默认保持 240×320、chunk 50、workers 0、ImageNet 初始化和原始夹爪标签。
审计警告确认参数仍在，所有指纹核验照常执行。不会自动启动短测后的正式训练，
不会连接机器人，不安装环境，也不会自动启动 W&B。

附加的训练参数覆盖脚本默认值；例如服务器只使用物理 GPU 1，另开一个新实验：

```bash
GPU_ID=1 bash scripts/wp_ws/train_flip_the_pillow_act.sh train \
  --run-name act_baseline_gpu1_30k_v2 --steps 30000 --batch-size 4
```

**这条命令是新训练，不是查看/恢复已有的 30000 步训练。已经在跑的训练不要重复启动。**
脚本只支持单 GPU；不会因为暴露多个设备就变成 DDP。默认 GPU 为单个
`CUDA_VISIBLE_DEVICES`（未设置则为 0），也可显式用 `GPU_ID` 选择。
解释器默认是当前环境的 `python`，可用 `PYTHON_BIN` 指定。已有运行名会拒绝覆盖；
需要另开实验时用 `--run-name` 改名。当前无断点续训入口。

## 11. 服务器已有训练：不依赖 W&B 看 loss

`src.wp_ws.view_loss` 只读已有训练的 `run_config.json / metrics.jsonl / 终态标记`，
不导入 torch、W&B 或机器人模块，不重启、不验证数据集、不更改训练参数。
文字查看仅需 Python 标准库。

先把新工具同步到服务器，在服务器项目根目录执行：

```bash
# 列出这台机器上的训练名字、请求步数、最近记录步数和结束状态。
python -m src.wp_ws.view_loss --list

# run-name 必须替换成上一步列出的实际训练名；以下是此前使用过的名字。
python -m src.wp_ws.view_loss --run-name act_baseline_gpu1_v1 --last 20 --follow

# 不跟随，只查看当前快照并导出 PNG；需可选的 matplotlib。
python -m src.wp_ws.view_loss --run-name act_baseline_gpu1_v1 --plot
```

文字查看先显示最近 20 条 train/val loss，以及最近/最佳验证值；`--follow`
再打印新记录，训练正常完成/失败后自动结束。Ctrl+C 只停查看器，不会停止训练。
如果没有结束标记，工具不会宣称训练进程仍然存活；需另查原终端/进程。
尚未写完的末尾一行会明确提示并暂不解析；不会修改/补造任何记录，跟随时等待写完。
显示精度为 6 位小数，不改变原始指标。

PNG 保存到中央路径：

```text
LOG_DIR/wp_ws/_loss_views/<task>/<run-name>/loss.png
```

图只包含实际记录的原始点，不补造中间 loss、不做平滑；横轴为训练 step。
该文件是**静态快照**，可用 VS Code Remote 等下载到本机打开；重复 `--plot` 更新它。
若同时用 `--plot --follow`，开始和结束跟随时更新图片，不是实时网页。
绘图缓存也限定在单独的输出目录，原训练目录、数据和 checkpoint 不变。
验证图注明原训练的完整验证/冒烟子集模式；查看器不能为正在运行的训练增加验证，
不能把子集 loss 转换成全量验证，也不把 normalized L1 当作成功率。

工具未同步时，仍可立即用现有服务器代码查看原始日志，不需要重启训练：

```bash
RUN_NAME=act_baseline_gpu1_v1  # 换成实际训练名
METRICS="$(python - "$RUN_NAME" <<'PY'
import sys
from src.config.paths import LOG_DIR
print(LOG_DIR / 'wp_ws' / 'flip_the_pillow' / sys.argv[1] / 'metrics.jsonl')
PY
)"
tail -n 20 -f "$METRICS"
```

这会显示 JSONL 原始记录；Ctrl+C 只结束 `tail`。W&B 网络恢复后，仍可从同一训练日志
补传曲线，无需重新训练。
