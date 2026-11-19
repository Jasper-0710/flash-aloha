# 翻转枕头：离线数据体检与固定划分

本目录提供**只读数据体检、显式启动的离线 ACT 训练及固定小样本诊断**。
体检/preflight 不训练；任何入口都不启动机器人、回放或采集。

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

## 12. 训练不稳定时：固定 32 个训练样本的可拟合性诊断

目的不是继续跑 100K，也不是把 train loss 改个名字当 val loss，而是先回答：
**在原数据加载/图像处理/原生 ACT 优化链路下，模型能否反复拟合一组固定输入？**
入口为 `src.wp_ws.overfit_act`，启动脚本为
`scripts/wp_ws/overfit_flip_the_pillow_act.sh`。训练仍由使用者执行。

### 实验控制与边界

- 默认取固定 train split 中排序最靠前的 **2 条 episode，各 16 个等距起始帧**，
  共 32 个输入；保存精确 episode/frame/flat index、有效 chunk 长度和缓存摘要。
  起始帧包含 episode 首尾；每个起点的最多 50 条未来标签可能与其他起点重叠。
  **不是 32 条独立轨迹，也不是 32 个单步标签。**
- 支持 `--episodes 0 1` 和 `--samples-per-episode 16` 显式指定；
  必须全部属于原训练 split。拒绝验证 episode、重复 ID 和不足数量的 episode，
  不通过复制样本凑数；不改写固定划分。
- 标准化仍由**原全部训练 episode**拟合，与完整训练保持相同口径，验证集不参与。
  不从 audit 的全数据摘要取统计量，也不改成小样本统计量。
- **复用现有 ActDataset** 的原始行配对、连续性边界、chunk/padding，以及
  `standard_image_processing`，没有第二套图像变换。夹爪超参考范围标签照常保留。
- **从新模型 + ImageNet 骨干初始化，不读取/续训旧 checkpoint**；模型、原生优化器、
  学习率、dropout、梯度裁剪保持原 baseline 默认设置。batch 仍为 16。
  只在这 32 个固定起点中跨 episode 混洗，不用旧完整训练的逐 episode 批采样。
  因此这是可拟合性诊断，**不是仅改变一个因素的全数据 sampler 消融**。
- 固定样本只解码一次并缓存于 CPU，默认图像缓存约 85 MiB；缓存上限默认 512 MiB。
  上限只针对解码后的数组，不包括临时 Parquet/Arrow 压缩 episode 或整个进程内存。
  原训练数据和审计仍要核验 SHA-256，准备阶段并非只读 32 行。
- step 0 和每 100 步在 `eval()` + `inference_mode()` 下重测**同一训练小样本**。
  训练阶段保持原 dropout；单个训练 batch 的 loss 与这里的全子集 eval 不应直接等同。
  聚合按有效 action 元素加权，padding 不计入。
- 不评估独立验证 episode、不连接硬件、不证明图像/动作同步正确、不测任务成功率。
  诊断 checkpoint 显式标注不可直接部署，无断点恢复入口。

### 本机运行

```bash
conda activate flash-aloha-act
cd /home/ubuntu/Desktop/code/VLA/flash-aloha

# 可选：只显示命令；不接触 CUDA、不下载、不写文件。
bash scripts/wp_ws/overfit_flip_the_pillow_act.sh train --dry-run

# 可选：CPU 核验及缓存测试；不构建/训练模型。
bash scripts/wp_ws/overfit_flip_the_pillow_act.sh preflight

# 实际小样本实验：1500 次更新，batch 16，仍为 chunk 50 / 240x320。
GPU_ID=0 bash scripts/wp_ws/overfit_flip_the_pillow_act.sh train
```

`train` 会自行执行 preflight，不要求先运行上面的独立 preflight。
所有同名 run 拒绝覆盖；第二次实验用 `--run-name act_overfit_32_local_v2`。
脚本不会安装依赖或自动登录/上传 W&B；缺少 ImageNet 权重缓存时，训练入口会由
原生 torchvision 下载，失败不会自动退化成 scratch。

直接 Python 命令等价为：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.wp_ws.overfit_act train \
  --run-name act_overfit_32_local_v1 \
  --samples-per-episode 16 --batch-size 16 \
  --steps 1500 --eval-every 100 --log-every 50 \
  --chunk-size 50 --height 240 --width 320 \
  --backbone-init imagenet --acknowledge-audit-warnings
```

服务器使用相同代码/数据/审计，在服务器的 Conda 环境和项目根目录执行：

```bash
GPU_ID=1 bash scripts/wp_ws/overfit_flip_the_pillow_act.sh train \
  --run-name act_overfit_32_gpu1_v1
```

单 GPU 足够做此诊断；当前入口不是 DDP，不要传 `GPU_ID=1,2,3`。
先不增大 batch、改图像尺寸或 chunk，避免在诊断时引入更多变化。

### 看什么结果

终端每次评估显示：

```text
[同训练样本 eval] step=... L1=... mean基线=... median基线=...
```

两个常数基线都在**同一组已选样本、相同标签权重**上计算：

1. `train_mean_constant`：始终预测原训练集动作均值，即标准化后的零向量。
2. `subset_median_constant`：把所选样本中的有效 chunk 标签汇总，拟合每维单一中位数；
   它是在这组目标上 L1 最优的单常数动作向量，不读取图像/state，不按 horizon 单独拟合。
   它比“零向量”更能排除模型只是拟合了这几条 episode 的常见姿态。

**不能拿之前全验证集上的 0.84279 均值基线直接作为这里的阈值**，样本集合不同。
判断同样本 eval 相比 step 0 是否持续、明显下降，以及是否明显优于这两个常数。
没有适用于所有动作的“loss < 某数就能部署”的阈值。

| 观测 | 能说明什么 / 下一步 |
|---|---|
| 同样本 eval 明显下降，并优于两个常数 | 这组样本的拟合链路基本可用；再做跨 episode 的全数据采样对照，用原固定验证集选择模型。不能证明相机同步、视觉依赖或任务成功。 |
| train batch loss 波动，但同样本 eval 明显下降 | 不要只盯训练的单 batch 点；继续看固定 eval 和逐维误差。 |
| 1500 步后同样本 eval 仍接近初始值/常数 | 先缩到 1 个输入验证优化链路，再分别检查输入、梯度更新和学习率等；不要直接开另一个 100K。不能仅据此断言数据错误。 |
| 关节误差下降但某侧夹爪/horizon 很差 | 先看逐维、逐 episode、逐 horizon 指标与原动作；不自动裁剪、平移动作或重配图像。 |

若需要进一步缩成一个固定输入（episode ID 应从 `selection.json` 选一个训练 ID）：

```bash
# 例：确认 episode 0 属于训练 split 后才使用 0。
GPU_ID=0 bash scripts/wp_ws/overfit_flip_the_pillow_act.sh train \
  --run-name act_overfit_one_sample_v1 --episodes 0 \
  --samples-per-episode 1 --batch-size 1 --steps 500 --eval-every 50
```

这只能测试一个输入的拟合，不构成完整数据修复或泛化结论。

### 输出与回传

通过中央 `src/config/paths.py` 写入：

```text
LOG_DIR/wp_ws/flip_the_pillow/<run-name>/
  selection.json            精确固定样本与缓存摘要
  normalization_stats.json  原训练 split 的统计量
  baselines.json            两个常数的同样本误差
  metrics.jsonl             overfit_train / overfit_eval；含逐维/episode/horizon 指标
  report.md                 每次评估后更新的简明表格
  run_config.json           配置、来源、实现摘要、评估边界
  COMPLETED.json            正常跑完后的摘要（中断/异常则 FAILED.json）
CKPT_DIR/wp_ws/flip_the_pillow/<run-name>/
  best.ckpt                 以同训练样本 eval 选择，可能是 step 0；不是验证集最佳
  last.ckpt                 最近评估点；含模型/统计/配置，不含优化器，不支持 resume
```

可将 `report.md` 和 `COMPLETED.json` 发回来；中断时发送已有 `report.md` / `FAILED.json`。
`COMPLETED` 只表示指定步数执行完，不是“已收敛”或机器人任务通过。
如覆盖了中央日志根目录，下面的读法仍适用：

```bash
python - <<'PY'
from src.config.paths import LOG_DIR
p = LOG_DIR / 'wp_ws' / 'flip_the_pillow' / 'act_overfit_32_local_v1'
print((p / 'report.md').read_text())
if (p / 'COMPLETED.json').exists():
    print((p / 'COMPLETED.json').read_text())
PY
```

**现有 `wandb_metrics` / `view_loss` 只支持常规 ACT 训练格式，本诊断未接入它们。**
不要改事件名冒充验证 loss；本轮先用终端、Markdown 表格和 JSON 指标判断。
合成 CPU 测试（需 ACT 环境的 torch，但不会启动 GPU 训练）：

```bash
python -m unittest src.wp_ws.tests.test_overfit_act -v
```

## 13. 小样本拟合通过后：全数据采样对照（先 10K，不增大 batch）

这个实验检验的是**采样顺序是否影响拟合/泛化**，不是预先认定旧 sampler 是根因。
32 个训练输入过拟合成功，不能把该组的 ~0.05 和 5823 个独立验证起点的 loss 直接比较。
先保持原模型、ImageNet 初始化、优化器/LR、batch 16、chunk 50、240×320、seed 42、
原 train/val split、原始夹爪标签和共享图像处理不变。两种模式都是**新初始化**，不续训旧 checkpoint。

### 单一对照变量与缓存边界

- `episode_block`：原来的按 episode 连续取 batch，CLI **默认仍保留这个行为**。
- `episode_window`：每个 epoch 打乱训练 episode，分成最多 8 个的窗口，在每个窗口内打乱所有起点，
  一个 batch 可以来自多个 episode。**不是全数据全局均匀乱序，也不保证每个 batch 都跨 episode**。
  每个 epoch 每个训练行恰好一次，不重复、不丢弃；窗口尾 batch 保留，batch 不跨窗口。
  每个样本内部的 action chunk、边界和 padding 完全沿用 `ActDataset.__getitem__`，绝不跨 episode 拼 action。
- 混采仅支持 `--workers 0`。采样器在读取该窗口的 batch 前声明窗口；窗口内每张 Arrow 表只加载一次，
  切换窗口释放旧表。**缓存的仍是含原始压缩图片字节的 Arrow 表，不是整套解码图像**。
- `--episode-cache-mib 1024` 限制当前窗口**保留的 Arrow buffer 总量**；超过会报错，
  不自动缩窗口或退化成频繁重读。此限额**不包括临时读取/解压、Python/Arrow 对象、验证单 episode、
  batch 张量、pinned memory、模型内存**，不是进程 RSS 或显存上限。
- 旧模式每个 episode 一个尾 batch，新模式每个窗口一个尾 batch，因此相同 step 的累计样本数可能略不同；
  日志记录 `samples_seen`、`batch_size_actual` 和 `batch_episode_count`，比较时应核对。
  固定 seed 不承诺 CUDA 逐位复现；单次对照也不足以证明因果。

### 固定评估，不再只盯单个 train batch

新启动脚本默认加上：

1. `--train-probe-per-episode 2`：每个训练 episode 在约 25% / 75% 的行位置各取一个原始输入，
   当前 91 个训练 episode 共 **182 个输入**，每次评估完全相同。保存精确行号与数组 SHA-256，
   只经现有适配器解码一次；默认 `--train-probe-cache-mib 512`，当前分辨率约 480.4 MiB 的 CPU 数组。
   这不是额外训练集/验证集，也没有改变训练起点的权重；它只是已参与训练的数据的固定测量点。
   每个 episode 同样选 2 点，所以**不等同于全训练集逐行 loss**。
2. `--eval-at-start`：更新前 step 0 评估；然后每 1000 步评估，最多先做 10000 步。
3. `--val-max-batches 0`：原 split 的 **全部 5823 个验证起点**，不抽 smoke subset、不重新划分。
   原模型 masked L1 按有效 action 元素累计；验证集不拟合任何标准化参数/参考向量。
4. 均值动作基线：始终预测训练集 mean（标准化后零向量），分别在 probe/验证的
   **同一批标签和 padding** 上计算。验证的参考向量不是在验证集上拟合的。
   probe 文件还保留该训练探针的单向量中位数基线，但不把它冒充验证成绩。
5. `best.ckpt` **只按验证 L1** 选，不按 train/probe 选；可能保留 step 0 的权重。
   `last.ckpt` 每 1000 步保存，仍无 resume 入口、仍不是可直接部署的模型。

### 用户执行

```bash
conda activate flash-aloha-act
cd /home/ubuntu/Desktop/code/VLA/flash-aloha

# 仅打印命令：不启动 Python/CUDA，不生成文件
bash scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh mixed --dry-run

# 可选：先只做 CPU 预检，检查指纹/标准化/混合 batch/固定探针
bash scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh preflight

# 建议先跑混采 10K，保持 bs16（会自动重复预检）
GPU_ID=0 bash scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh mixed
```

独立日志/ckpt 名分别为 `act_mix8_preflight_v1` 和 `act_mix8_b16_10k_v1`；已存在会拒绝覆盖。
重试时显式给新的 `--run-name`。服务器复制本次代码后：激活服务器的 `flash-aloha`，
进入 `/data1/wp_ws/flash-aloha`，使用 `GPU_ID=1`（或另一张获准的单卡）。
RTX 6000 PRO 的显存更大不意味着本次要改 batch；这个入口也不是 6 卡 DDP。

需要严格的新对照时单独执行（不要由同一张卡并发两项）：

```bash
GPU_ID=0 bash scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh control
```

对照 run 为 `act_block_b16_10k_v1`。两个 launcher 配置只在 sampler/run-name 上不同；
旧 100K 实验的 5K/10K 验证点可参考，但缺失相同的固定 probe 和 step-0 指标，不能代替完整对照。
不要为了等 10K 结束而忽略已经出现的异常；OOM/非有限 loss/缓存超限应停下看错误，勿直接重开 100K。

### 看 loss 与回传

```bash
# 不依赖 W&B 网络，包含训练探针和验证；添加 --plot 可导出本地 PNG
python -m src.wp_ws.view_loss --run-name act_mix8_b16_10k_v1 --follow

# 另一终端；仅上传白名单标量，不上传图片/动作/模型。403 不影响训练，用上面的本地入口即可。
python -m src.wp_ws.wandb_metrics \
  --run-name act_mix8_b16_10k_v1 --project flash-aloha --mode online --follow
```

重点曲线：

| 指标 | 含义 |
|---|---|
| `train/normalized_l1` | 原日志：某一次训练 batch 的 L1，含训练时 dropout，不同点的输入不同 |
| `train/window_normalized_l1` | 从上次日志到本次日志的有效元素加权平均；跨更新的描述量，不是同一权重的固定集评估 |
| `train_probe/normalized_l1` | 固定 182 个训练输入，eval/no-grad 下测量拟合趋势，**不是 held-out validation** |
| `val/normalized_l1` | 固定完整验证集，才用它选择 best checkpoint |
| `val/mean_constant_normalized_l1` | 同一验证集的均值动作参考线；持续、明显优于它比只看训练 loss 更有意义 |
| `train/batch_episode_count` | 当前 batch 里有几个 episode，检查混采是否真的生效 |

新训练在每次评估后更新：

```text
LOG_DIR/wp_ws/flip_the_pillow/act_mix8_b16_10k_v1/
  report.md                     step / 固定 train probe / 完整验证 / 各自均值参考线
  train_probe_selection.json    精确训练起点、缓存数组摘要与选择算法
  train_probe_baselines.json    probe 的两个同样本常数参考
  metrics.jsonl                 train / train_probe / validation，probe 含逐维/episode/horizon 误差
  run_config.json               混采/缓存/评估配置及来源
  COMPLETED.json / FAILED.json  运行状态（不是任务成功/失败的评估）
```

跑到 3K–5K 可以先看趋势，不用等 10K 才回传 `report.md`；保留最后的 `COMPLETED.json`/`FAILED.json`。
训练 probe 降、验证降且持续优于常数，才考虑延长；probe 降而验证停滞/变差，要研究泛化差距；
两者都不降，进一步排查而非只增加 step 或 batch。10K 仅是第一轮诊断预算，不是必然收敛的期限。
这些都是离线信号，不证明视觉依赖、采集曝光同步、真机动作成功或硬件安全。

合成 CPU 测试与脚本语法检查（不启动真实训练/机器人）：

```bash
python -m unittest discover -s src/wp_ws/tests -v
bash -n scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh
```

## 14. 完整验证集离线评估：动作维度、预测距离、逐 episode 对比

完成 `act_mix8_b16_10k_v1` 后，下一步先看 **best checkpoint 的完整固定验证集误差和动作曲线**，
不直接增加 step/batch 或上机器人。此工具不训练、不下载模型、不写原数据/训练日志/checkpoint，
也不重算统计、不重新划分数据、不变更 action 对齐、夹爪、图像或 chunk masking。

### 用户执行

```bash
conda activate flash-aloha-act
cd /home/ubuntu/Desktop/code/VLA/flash-aloha

# 可先确认命令：不启动 Python，不访问 CUDA、不生成文件
bash scripts/wp_ws/evaluate_flip_the_pillow_act.sh evaluate --dry-run

# CPU 预检：读 checkpoint/核对指纹/解码一个验证 batch，不做模型推理
bash scripts/wp_ws/evaluate_flip_the_pillow_act.sh preflight

# 预检通过后：单卡离线评估，自动再次核对；默认 batch 与原验证一致（这轮为 16）
GPU_ID=0 bash scripts/wp_ws/evaluate_flip_the_pillow_act.sh evaluate
```

默认来源：`CKPT_DIR/wp_ws/flip_the_pillow/act_mix8_b16_10k_v1/best.ckpt`；
同时要求同名的训练日志及其 `COMPLETED.json`。这不是部署入口，不接受任意外部 checkpoint。
使用 restricted `torch.load(weights_only=True, map_location="cpu")`，失败时不退回不受限 pickle。
CPU 预检验证文件内容/声明和数据适配器；**原生模型参数形状的严格加载及 GPU 推理仍在 evaluate 阶段完成**。
只加载本项目自己产生、来源可信的模型，文件哈希证明一致性而不是外部真实性。

默认 CPU 预检输出名为 `act_mix8_eval_preflight_v1`，正式评估输出名为 `act_mix8_eval_v1`。
已存在就拒绝覆盖；重试指定新名字，例如 `--run-name act_mix8_eval_v2`。
可用 `--source-run OTHER_RUN --checkpoint best` 指向另一轮**已完整完成且用完整固定验证集**的 baseline；
`--checkpoint last` 要求其 step 和最后验证记录完全一致。中断、正在训练、smoke subset、
`act_overfit` 格式和 RTC checkpoint 不会被这个入口当作可比较的结果。

服务器：同步本次代码，进入 `/data1/wp_ws/flash-aloha`，激活 `flash-aloha`，使用 `GPU_ID=1`（或另一张获准单卡）。
来源 run 必须在**执行机器**对应的中央日志/ckpt 路径下；本机存在不代表服务器上有。
不要为了凑齐目录伪造 `COMPLETED.json` 或 audit。这里不支持六卡 DDP。
原数据、audit、训练记录及其指纹须相互匹配；若跨机器搬运，应保留对应产物，不能悄悄改统计或配对。

等价模块命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.wp_ws.evaluate_act evaluate \
  --source-run act_mix8_b16_10k_v1 --checkpoint best \
  --run-name act_mix8_eval_v1 --examples-per-episode 2
```

没有图像尺寸、chunk、action shift、统计或验证划分覆盖选项：它们全部来自原 checkpoint/固定 audit。
可显式改 `--batch-size`，但会记录与原验证的差异；**首轮先不改**。
不会调用 ImageNet 权重下载、优化器或反向传播。

### 做哪些检查

1. 核对 checkpoint 与日志/ckpt 目录的配置、标准化文件、完成记录、保存 step。
2. 重验源数据 SHA-256/audit 指纹，确认原来的验证 episode、continuity 和 provenance 未变；
   不用验证数据拟合任何统计量。相关推理实现和共享图像处理发生改变时拒绝静默迁移。
3. 复用 `ActDataset` 和原 `EpisodeBatchSampler(shuffle=False)`；每个验证起点恰好一次，
   使用原图像处理和动作 padding，首个目标仍是同一保存行的 action，绝不读取未来动作来辅助预测。
4. `eval()` + `inference_mode()` 逐 episode 推理；模型 L1 和训练均值常数 L1 都按有效 action 元素累计，
   不做 batch 均值的无权重平均。指标有总体、逐 episode、逐维度、逐 horizon 四种视角。
5. 与 checkpoint 保存的验证 L1 复现对照：数量/目标基线不一致直接失败；
   模型 L1 偏差超过 `0.0005 + 0.005 × 原 L1` 标为 `REPRODUCTION_WARNING`，先核对环境/批大小，
   不自动解释为性能变化。**此容差不是任务成功或硬件安全阈值**。
6. 结束前再次检查输入未变；所有产物成功后才写新输出目录的 `COMPLETED.json`。
   失败只留下新目录里的 `FAILED.json` 和部分诊断，不能把部分 PNG 当作完整评估。

### 输出和如何看

```text
LOG_DIR/wp_ws/flip_the_pillow/act_mix8_eval_v1/
  run_config.json               来源 checkpoint SHA-256、代码/预处理/数据证据、评估配置
  preflight.json                CPU 预检，不是 GPU/动作成功证明
  examples.json                 看结果前选定的每条 episode 的等分中点（默认 25%、75%）
  metrics.json                  总体、逐维、逐 horizon、逐 episode、复现结果
  report.md                     中文摘要和单位说明
  predictions/episode_XXXXXX.npz 全部标准化预测/目标、padding、frame index、train mean/std、连续区间
  plots/horizon_errors.png      offset 0…H−1：L1、关节度数、夹爪原单位误差和有效数
  plots/episode_errors.png      每条验证 episode 与训练均值常数比较
  plots/t0_episode_XXXXXX.png    每个真实观测重新预测的第 0 个动作，不是 rollout
  plots/chunk_episode_…_frame_….png  单个固定观测预测的完整有效 chunk
  COMPLETED.json / FAILED.json  评估程序状态，不是翻转枕头成功/失败
```

- **先看复现**：本轮原验证 L1 是约 `0.397504`，均值常数约 `0.842791`。
  这是原训练记录，**不是本工具已经测出的新结果**。不能拿 `train_probe` 的误差冒充验证指标。
- **再看泛化分布**：是不是所有验证 episode 都优于简单常数，还是少数样本拖累？
  常数是很弱的参照，优于它不证明学会了视觉或接触控制。
- **再看动作时序**：关节接近、翻转和夹爪开合的位置/时刻是否跟上目标，是否预测成平滑均值、
  转折错位或单侧偏差。不要只看全维度平均 loss。
- **看预测距离时同时看有效数**：后段 chunk 有 padding，有效样本不相同；
  同一动作也会出现在多个重叠 chunk，不是去重后的单帧评价。
- 图按 checkpoint 统计量反标准化，关节单位 rad，报告另给 degree；夹爪保留原归一化单位，
  **不裁剪 `[0,1]`、不改成电机角度**。反标准化存在 float32 舍入。
- 横轴是已保存 frame index/预测 offset，不按 FPS 伪造相机时间戳；不跨 continuity 边界连轨迹线。
- `.npz` 用 `np.load(path, allow_pickle=False)` 读取，不包含图片或 Python 对象。该工具不自动上传 W&B。

**验证集已参与 best checkpoint 选择，不是新的独立测试集；离线预测仍以真实录制状态和图像为输入，
不能证明闭环稳定性、视觉依赖、采集曝光同步、控制时延或真机任务成功率。**
先回传新 `report.md`；必要时再结合逐 episode/夹爪曲线决定延长训练、收集数据或设计下一项离线对照。
不要仅靠进一步压低这份验证 loss 就认定可部署。

合成 CPU 测试（不会读真实录制数据或启动 GPU/机器人）：

```bash
python -m unittest src.wp_ws.tests.test_evaluate_act src.wp_ws.tests.test_evaluation_launcher -v
python -m unittest discover -s src/wp_ws/tests -v
bash -n scripts/wp_ws/evaluate_flip_the_pillow_act.sh
```
