# 任务四复现说明

## 从头训练

1. 在 Kaggle 中创建 Notebook，挂载比赛 `2026task-4` 官方数据，选择 GPU T4，开启 Internet 以安装依赖及下载 ImageNet 预训练权重。
2. 导入 `task4_training.ipynb`，使用 Save Version → Save & Run All 完整执行。
3. 程序会自行核验官方转换函数、按完整图像划分四折、训练两种模型并进行全量拟合，输出检查点、参数选择、预测及 CSV。
4. 以 `task4_results.json` 的候选文件名和 SHA256 为准；实际公开榜得分以 `submission_results.json` 为准。本次最终公开榜最高分为 0.84867。

仅使用官方 20 张训练标签和 ImageNet 预训练；不需要下载任何外部眼底数据或测试标签。

## 使用已保存权重复现预测

1. 将已保存的输出或解压后的 `task4_results.zip` 挂载到 Kaggle。该目录应包含 `models/` 和 `task4_manifest.json`。使用清单核对权重，避免混入其他实验的模型。
2. 同时挂载任务四官方数据，导入相同 Notebook。在最后的运行单元前设置：

   ```python
   import os
   os.environ['TASK4_REUSE_DIR'] = '/kaggle/input/你的已保存输出目录'
   ```

3. 运行 Notebook。程序将相应 `.pt` 文件复制到工作目录并跳过已存在检查点的训练，重新生成验证预测、参数选择和测试预测。完整复用需要两种架构各四个折模型及一个全量模型，共十个检查点。缺少的检查点会按源码重新训练。
4. 推荐使用原 Kaggle GPU 镜像和源码中锁定的依赖版本。实际环境版本、种子、划分、训练图像列表、检查点最佳轮数都保存在结果或检查点中。不同硬件及库版本可能导致浮点数细微差异。

## 检查最终 CSV

使用带 NumPy 与 Pillow 的 Python 环境执行：

```sh
python verify_submission.py submission_final.csv
```

检查器验证列名、20 行 ID 顺序、512×512 范围、正数游程、排序与不重叠，并生成掩码和眼底图的可视化核验图。它只检查与展示预测，不会修改预测，也不需要测试真值。

比赛网页文字和实际标签的黑白方向不同，必须遵循实际数据与官方转换程序。官方程序编码的是缩放后所有正值的白色像素，按行优先排列、从 1 开始，不能改为按列展平。

## 验证分数的适用范围

OOF 图像与对应训练折分离，但同一验证图像被用于选择检查点、阈值、增强和融合设置，所以该分数是调参过程中的估计。全量模型已经见过全部 20 张训练标签，不能再对这些图像计算分数并称其为独立验证。公开榜只使用约一半测试数据，最终隐藏榜可能改变模型排序。


## 复现最终增强方案

1. 先获得主训练 Version 1（ID 364133019）的完整输出，或使用本地 `cloud_artifacts/`；其中需要原始源代码、十个检查点、每架构保存的四方向验证和测试概率，以及主结果 JSON。单独挂载检查点不足以运行增强脚本。
2. 创建 Kaggle Notebook，同时挂载官方比赛数据及自己的主训练 Notebook 输出。导入 `task4_refinement.ipynb`，选择 GPU，使用锁定依赖完整执行。`refine_task4.py` 会自动查找输入，核对全部模型哈希，并保留原划分。
3. 脚本比较 512 与 768 八方向推理和两尺度平均，按留出图像选择架构比例、阈值和小连通域。它不会训练模型，也不会下载外部眼底标签。
4. 本次选中 `multiscale_d4`、第二架构权重 0.5、阈值 0.35、最小连通域 24。最高分对应 `submission_task4_refined_folds.csv`，即每架构四个折模型平均后，两架构各 50%；不加入全量模型。
5. `submission_final.csv` 与该文件逐字节相同。复现时核对 `task4_refinement_results.json` 和 `task4_refinement_manifest.json`；硬件或依赖改变时预测可能有微小数值差异。已保存的原始 CSV 可通过 SHA256 精确比对。

本地 `cloud_backup_verification.json` 与 `refinement_backup_verification.json` 是备份核验记录；`status.json` 保存最终候选及检查时排名。源码、数据、模型及概率文件都留在 `task4/` 内，旧项目和历史会话未修改。
