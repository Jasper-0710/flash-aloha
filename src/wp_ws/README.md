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
