# 机器学习课程设计复查代码

四个目录按报告章次排列。每项实验使用对应 Kaggle 比赛的官方数据。训练请使用支持 CUDA 的 Kaggle Notebook 并挂载本人有权限使用的比赛数据。原始脚本与 Notebook 保留实验时的路径约定。历史 CSV 与 JSON 供核对过程和结果。SHA256.json 记录这些归档文件的哈希。

| 章节 | Kaggle 数据 | 入口及复查材料 |
| --- | --- | --- |
| 1 植物图像分类 | [2026task1](https://www.kaggle.com/competitions/2026task1) | task1/plant_efficientnet_kaggle.ipynb；train_plant_efficientnet.py 训练并保存各折模型，select_plant_tta.py 继续选择推理方案；kaggle_result.json 记录提交版本和哈希 |
| 2 人脸表情分类 | [2026task2](https://www.kaggle.com/competitions/2026task2) | task2/task2_fer_kaggle.ipynb 运行历史基线；再挂载原版本 357223684 的输出，执行 task2_optimization_20261018.ipynb 中的六个优化单元；experiment_results.json 与 history/ 用于核对 |
| 3 眼底中心凹定位 | [2026task3](https://www.kaggle.com/competitions/2026task3) | task3/task3_training.ipynb 运行分组四折粗细热图模型；run_local.py 为两组 MPS 模型的本机流程；task3_detector_cloud.ipynb 运行检测模型热启动、全量拟合和最终融合 |
| 4 眼底血管分割 | [2026task-4](https://www.kaggle.com/competitions/2026task-4) | task4/task4_training.ipynb 训练十个检查点；挂载主 Notebook 输出后运行 task4_refinement.ipynb；详见 task4/REPRODUCE.md |

任务 2 的优化 Notebook 需要原流程的权重、固定验证索引和已有预测。历史输出版本为 [357223684](https://www.kaggle.com/code/shabiba2026/2026task2-fer-deep-learning?scriptVersionId=357223684)，最终输出版本为 [359730229](https://www.kaggle.com/code/shabiba2026/2026task2-fer-deep-learning/output?scriptVersionId=359730229)。依赖包括 PyTorch、torchvision、scikit-learn、OpenCV 和 facenet-pytorch 2.6.0。

任务 3 先用 train_detector.py 生成首折检测模型检查点，并以 evaluate_plain.py、robust_ensemble.py、geometric_ensemble.py 生成稳健热图候选。最终 [361421000](https://www.kaggle.com/code/shabiba2026/2026task3-cloud-detector-optimization?scriptVersionId=361421000) 还依赖原私有恢复包中的折索引、第 55/56 轮检测权重、同一留出集热图预测和稳健候选。见 optimize_detector_cloud.py 的输入检查与输出文件名。代码仓库供复查，逐字节复现该次运行需要原私有输出或重新生成相应中间产物。

任务 4 的主训练版本为 364133019，增强版本为 364147975。官方数据源提供转换器 segmentation_to_csv.py。主输出包含十个检查点和四向预测概率，增强流程复用它们。

仓库包含实验源码、源 Notebook、训练指标及复查说明。比赛图像、官方标签、模型权重和私有预测文件由比赛和原实验账户分别保存。
