# Unsupervised Video Anomaly Detection（US-VAD）

## Multi-task (2021)

> 标题：**Anomaly Detection in Video via Self-Supervised and Multi-Task Learning**
>
> 链接：https://arxiv.org/abs/2011.07491
>
> 解读：https://zhuanlan.zhihu.com/p/386810871
>
> 关键词：Unsupervised（无监督），Multi-task Learning（多任务学习），Knowledge Distillation（知识蒸馏）
>
> 对我们的价值：如果我们要设计自监督任务，可以参考这个工作里面设计的任务

### 框架图

![image-20220415221005407](D:\CodeField\Notebook\Pictures.assets\image-20220415221005407.png)

### 描述

首先，使用一个预训练的YOLOv3对每一帧的物体进行检测，得到一系列的bounding boxes。对每帧检测到的物体，生成一个物体为中心的时间序列 object-centric temporal sequence（具体操作是裁剪物体前后序列帧的bounding box对应范围，缩放到64x64的大小）。生成的时间序列作为3D CNN的输入，结构由一个共享的3D CNN和对应四个预测任务的四个分支组成。**推断的时候，平均四个任务的分数得到最终的异常分数**。

三个自监督任务是：1.区分向前/向后移动的物体（时间箭头）；2.连续/间歇帧中物体的区分（运动不规则）；3.物体特定外观信息的重建。

一个知识蒸馏任务是：YOLOV3和ResNet50都是teacher network，特征提取网络(3D 卷积) 是student network。希望3D卷积网络：1. 特征提取性能接近ResNet50；2. 遇到正常物体(如人行道行人)与YOLOV3检测结果差异小，遇到异常物体(如人行道车辆)与YOLOV3检测结果差异大。

## Hybrid (2021)

> 标题：**A Hybrid Video Anomaly Detection Framework via Memory-Augmented Flow Reconstruction and Flow-Guided Frame Prediction**
>
> 链接：https://arxiv.org/pdf/2108.06852.pdf
>
> 解读：https://www.cnblogs.com/Cucucudeblog/p/15573534.html
>
> 关键词：Unsupervised（无监督），Auto-encoder（自编码器），Reconstruction（重构），Optical flow（光流）
>
> 对我们的价值：目前没太大价值，但如果想利用光流，可以参考他们设计的重构误差

### 框架图

![image-20220415230400111](D:\CodeField\Notebook\Pictures.assets\image-20220415230400111.png)

### 描述

由两部分构成：1. 多级记忆增强自动编码器（左下，绿色），用于光流重建：采用多个记忆模块来记忆不同特征级别的正常模式，同时在编码器和解码器之间添加跳过连接，以**补偿由于记忆而造成的强信息压缩**；2. 条件变分自动编码器（右上，蓝色），同时接受原始视频帧和光流帧作为输入，用于预测未来帧：一方面，以**绿色编码器的重构流作为条件**，**将重构条件统一到预测管道**中；另一方面，**最大化观察视频帧和重构光流帧变量诱导的证据下界(ELBO)量**，对输入帧和未来帧流的一致性进行编码。

整个框架只使用正常数据进行训练；在测试阶段，根据重构得到的光流和预测得到的帧计算误差来判断异常情况：重构的**正常流质量较高**，通过预测模块可以成功地预测未来的帧，预测误差较小；而重构的**异常流质量较低**，从而导致未来帧的预测误差较大。

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Weakly Supervised Video Anomaly Detection（WS-VAD）

## MIST (2021)

> 标题：**MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection**
>
> 链接：https://arxiv.org/abs/2104.01633
>
> 解读：https://blog.csdn.net/qq_45496282/article/details/120491135
>
> 关键词：Weakly supervised（弱监督），Multiple Instance Learning（多实例学习），Self-training（自训练）， Pseudo Label（伪标签），Self-Guided Attention（自引导注意）
>
> 对我们的价值：帧级别的伪标签生成；自引导注意力模块，能够关注到每帧中的异常区域

### 框架图

![image-20220415222744444](D:\CodeField\Notebook\Pictures.assets\image-20220415222744444.png)

### 描述

MIST由两部分组成：1. 多实例伪标签生成器，该生成器采用稀疏连续采样策略，为anomalous video产生更可靠的clip-level伪标签；2. self-guided attention，该encoder目的是自动关注帧中的异常区域，同时提取特定的任务表示。

以下为整体训练方案：

<img src="D:\CodeField\Notebook\Pictures.assets\image-20220415222345552.png" alt="image-20220415222345552" style="zoom:80%;" />

以下为伪标签生成器：

<img src="D:\CodeField\Notebook\Pictures.assets\image-20220415224513722.png" alt="image-20220415224513722" style="zoom:80%;" />

以下为自引导注意力模块：

![image-20220415224630179](D:\CodeField\Notebook\Pictures.assets\image-20220415224630179.png)

作者采用**自训练方案**优化两个组件。对于normal video，无需伪标签，因为我们知道一定是每一个clip都是normal；对于abnormal video，用生成的伪标签作为标签。

两个loss：1. ranking loss：最大化“正常视频bag的最高sub-bag score”和“异常视频bag的最高sub-bag score”；2. classification loss：最小化分类结果与伪标签的差距

## WSAL (2021)

> 标题：**Localizing Anomalies From Weakly-Labeled Videos**
>
> 链接：https://arxiv.org/pdf/2008.08944.pdf
>
> 解读：无
>
> 关键词：Weakly supervised（弱监督），Localization（定位）
>
> 对我们的价值：捕获时序上的上下文信息；定位异常帧；降噪技术

### 框架图

![image-20220417001436650](D:\CodeField\Notebook\Pictures.assets\image-20220417001436650.png)

### 描述

依然是先将视频切割为等长片段，然后挨个送入HCE模块；每个片段都会有一个对应的anomaly score。而后联合降噪和伪造异常模块，做MIL损失。

降噪：训练阶段给视频加入噪声，实际上并没有降噪，而是想减小模型对噪声的敏感度

伪造异常：因为缺乏帧级别标注，想通过切割、拼接等操作人为伪造这样的异常训练数据，个人认为不靠谱

## CRFD (2021)

> 标题：**Learning Causal Temporal Relation and Feature Discrimination for Anomaly Detection**
>
> 链接：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9369126
>
> 解读：无
>
> 关键词：Weakly supervised（弱监督），Localization（定位）
>
> 对我们的价值：捕获时序上的上下文信息；增强特征的区分度

### 框架图

![image-20220417100518854](D:\CodeField\Notebook\Pictures.assets\image-20220417100518854.png)

### 描述

从18年Real-World那一篇开始，现有弱监督视频异常检测的方法基本上都是在MIL（Multiple Instance Learning）上做文章，但是有两个因素从来没有被考虑过：

1. **时序关系**：之前的工作只在每一个视频片段做文章，忽略了片段与片段之间的时序关系；有的工作尽管考虑了片段间的时序关系，但是只限于训练阶段，测试阶段还是没有利用这一关系（由于在测试阶段，未来帧不可知）；
2. **feature的区分度**：之前的工作基本都用MIL方法，能确保类间可区分，但无法确保类间的离散程度和类内的紧致程度。

于是想解决这两个问题：

1. 提出**CTR（causal temporal）**和**CS（classifier）模块**，用以捕获视频片段之间的时序信息。

   ![image-20220417175005758](D:\CodeField\Notebook\Pictures.assets\image-20220417175005758.png)

   用了temporal attention mechanism，把历史帧的信息融合入当前帧。

   ![image-20220417175216204](D:\CodeField\Notebook\Pictures.assets\image-20220417175216204.png)

   用bag里top-k的均值和01标签做交叉熵。

2. 提出**CP（compactness）**和**DP（dispersion）模块**，用以增强feature的类间区分度。

   ![image-20220417174715656](D:\CodeField\Notebook\Pictures.assets\image-20220417174715656.png)

​		CP：想把normal video的features拉近

​		DP：想把abnormal video中的normal片段的features和normal video的features拉近，把abnormal 				 video中的abnormal 片段的features和normal video的features拉远

## RTFM (2021)

> 标题：**Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning**
>
> 链接：https://arxiv.org/pdf/2101.10030.pdf
>
> 解读：https://www.cnblogs.com/lhiker/articles/15599788.html
>
> 关键词：Weakly supervised（弱监督），Multiple Instance Learning（多实例学习）
>
> 对我们的价值：对MIL loss做了改进

### 框架图

![image-20220417175816322](D:\CodeField\Notebook\Pictures.assets\image-20220417175816322.png)

### 描述

由于MIL选取max的特性，如果一开始模型就选取了一个本身为正常的视频片段并给予了最高anomaly score，这一误差会在后续训练被放大；另外，异常事件往往发生于多个的视频片段，我们会失去让模型在这些视频上学习的机会。

基于上述想法，作者将MIL改进为用top-k的均值做loss。

此外，作者还提出用feature模长而非anomaly score做指标。用的什么feature？用Multi-scale Temporal Network（MTN）生成的feature。

![image-20220421233208321](D:\CodeField\Notebook\Pictures.assets\image-20220421233208321.png)

## MSL (2022)

> 标题：**Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection**
>
> 链接：https://www.aaai.org/AAAI22Papers/AAAI-6637.LiS.pdf
>
> 解读：无
>
> 关键词：Weakly supervised（弱监督）
>
> 对我们的价值：对MIL loss做了改进；视频片段级别的伪标签生成

### 框架图

![image-20220419130439904](D:\CodeField\Notebook\Pictures.assets\image-20220419130439904.png)

### 描述

由于MIL选取max的特性，如果一开始模型就选取了一个本身为正常的视频片段并给予了最高anomaly score，这一误差会在后续训练被放大；另外，异常事件往往横跨多个连续的视频片段。

基于上述想法，作者将MIL改进为MSL: multiple sequence learning。不再关注video bag中分数最高的单个视频片段，而是关注video bag中分数最高的由多个视频片段连接而成的视频序列。

此外，作者还提出双阶段自训练框架。不断地迭代以下过程：生成伪标签—用伪标签训练—用预测值训练—再生成伪标签—…

