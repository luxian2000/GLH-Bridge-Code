# GLH-Bridge-Code 模型结构详解

## 1. 仓库模型体系总览

本仓库基于 MMRotate，包含两类模型实现：

- 标准旋转检测器（来自 MMRotate 主线）
- 面向大幅面桥梁检测的 ImgFPN 扩展检测器（仓库核心改造）

在桥梁基准配置目录中，主要模型家族如下：

| 配置文件 | 顶层检测器类型 | 结构范式 |
|---|---|---|
| configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_allbridge_oc.py | OrientedRCNN | 两阶段 |
| configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py | RotatedTwoStageDetectorImgFPN2 | 两阶段 + 全局局部融合 |
| configs/bridge_benchmark/roi_trans_r50_fpn_2x_allbridgelocal_oc.py | RoITransformer | 两阶段 |
| configs/bridge_benchmark/redet_re50_refpn_2x_allbridge_oc.py | ReDet | 两阶段 |
| configs/bridge_benchmark/rotated_faster_rcnn_r50_fpn_2x_allbridge_oc.py | RotatedFasterRCNN | 两阶段 |
| configs/bridge_benchmark/r3det_r50_fpn_2x_allbridge_oc.py | R3Det | 单阶段（精修） |
| configs/bridge_benchmark/oriented_reppoints_r50_fpn_2x_allbridgelocal_oc.py | RotatedRepPoints | 单阶段 |
| configs/bridge_benchmark/rotated_fcos_r50_fpn_2x_allbridge_oc.py | RotatedFCOS | 单阶段 |
| configs/bridge_benchmark/rotated_fcos_r50_fpn_2x_allbridge_ImgFPN_oc.py | RotatedSingleStageDetectorImgFPN | 单阶段 + 全局局部融合 |
| configs/bridge_benchmark/rotated_retinanet_obb_r50_fpn_2x_allbridge_oc.py | RotatedRetinaNet | 单阶段 |
| configs/bridge_benchmark/rotated_retinanet_obb_r50_fpn_2x_allbridge_le90.py | RotatedRetinaNet | 单阶段 |

说明：

- r3det_kld_r50_fpn_2x_allbridge_oc.py 是在 R3Det 基础上覆盖损失项（KLD/GDLoss），属于“继承配置 + 改损失”而非新增顶层检测器。
- train_dist_mmrot.sh 默认指向 oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py，因此该仓库主训练入口是 ImgFPN 两阶段版本。


## 2. 主模型：RotatedTwoStageDetectorImgFPN2（HBD-Net 实现主线）

对应配置：

- configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py

对应实现：

- mmrotate/models/detectors/two_stage_ImgFPN.py
- mmrotate/models/detectors/img_split_bridge_tools.py
- mmrotate/datasets/pipelines/loading.py
- mmrotate/datasets/pipelines/transforms.py


## 3. 结构分解（模块级）

### 3.1 参数化主干

局部分支和全局分支各自维护一套同构网络：

- local backbone: ResNet-50
- local neck: FPN (P2-P6, out_channels=256)
- global backbone: ResNet-50
- global neck: FPN (P2-P6, out_channels=256)

此外还构建两套候选框/检测头：

- local rpn_head + local roi_head
- global rpn_head + global_roi_head

这使得模型可同时学习：

- 局部高分辨率 patch 的细粒度定位
- 低分辨率全局图的大尺度上下文


### 3.2 FPN 融合模块（关键改造）

文件：mmrotate/models/detectors/two_stage_ImgFPN.py

检测器初始化时创建 fusion_convs（ModuleList）：

- 第 k 个融合卷积输入通道为 (k+2) * 256，输出 256
- 卷积核 3x3，padding=1

融合逻辑 ConcateFPNFea：

1. 对每个全局尺度，根据 rel_x0y0x1y1 把与当前 local patch 对齐的全局特征区域裁出来。
2. 对每个 local FPN 层，尝试拼接来自更高语义层的全局特征。
3. 按拼接层数选择对应 fusion_convs 做通道压缩，得到融合后的 local 特征。

本质上是“局部特征为主、全局语义补充”的跨尺度拼接融合。


### 3.3 数据管线中的全局信息构造

配置中使用了自定义流水线：

- LoadFPNImageFromFile
- RandomFlipImgFPN
- NormalizeImgFPN

其中 LoadFPNImageFromFile 负责额外输出：

- g_img_list: 多尺度全局图块列表
- g_img_infos: 对应元数据（down_ratio、rel_x0y0x1y1、gt_box、labels）

这些字段会在 Collect 阶段与 local img 一起送入 forward_train。


## 4. 训练前向流程（forward_train）

入口：RotatedTwoStageDetectorImgFPN2.forward_train

### 4.1 Local 分支

1. local 图像经过 local backbone + local neck 得到 l_fea。
2. 若存在可用 global 输入，先融合成 merge_l_fea；否则 merge_l_fea=l_fea。
3. local RPN 在 merge_l_fea 上计算 proposal 与 rpn loss。
4. local RoIHead 在 merge_l_fea 上计算 cls/reg loss。


### 4.2 Global 分支（可选）

当 g_img_list 非空时：

1. Collect_Global 整理各 global 层 gt_box/labels，并补充 img_shape、pad_shape、scale_factor。
2. filter_small_ann 按长度阈值过滤较小目标（代码中默认 length_thr=15）。
3. global backbone+neck 提取 g_fea。
4. global RPN 与 global RoIHead 独立计算 global loss。
5. 这些 loss 以 *_global 命名并加入总 loss。


### 4.3 总损失组成

总体包含：

- local: loss_rpn_cls, loss_rpn_bbox, loss_cls, loss_bbox, acc
- global: loss_rpn_cls_global, loss_rpn_bbox_global, loss_cls_global, loss_bbox_global, acc_global

当 global 输入为空时，global loss 分量被置零张量，保证训练图结构稳定。


## 5. 推理流程（simple_test）

入口：RotatedTwoStageDetectorImgFPN2.simple_test

### 5.1 金字塔尺度生成

从原图宽高开始不断除 2，直到不大于 1024，得到最小全局尺度（global_shape_min），并记录中间尺度列表。


### 5.2 最小全局层推理

1. 先在最小 global 图上做一次完整 global 检测。
2. 检测框按缩放比例映射回原图坐标。


### 5.3 其余尺度 patch 化推理

对每个较大 global 尺度：

1. 先缩放到该尺度。
2. FullImageCrop 以 patch_shape=(1024,1024), gaps=[200] 滑窗切块。
3. 每批 patch 经过 global 分支推理。
4. patch 检测结果 relocate 到大图，再按 scale 放大到原图。
5. merge_results_two_stage 对当前尺度内部做旋转 NMS 聚合。


### 5.4 原始分辨率 local patch 推理

对原图同样滑窗切块：

1. local 分支推理每个 patch。
2. relocate 回原图坐标。
3. merge_results_two_stage 做 local 内部聚合。


### 5.5 最终融合

将“多尺度 global 结果 + local 结果”拼接后调用 merge_results_tensor 统一做旋转 NMS，得到最终输出。


## 6. 框级操作与坐标系处理

工具文件：mmrotate/models/detectors/img_split_bridge_tools.py

关键函数：

- get_sliding_window: 按 size/gap 生成滑窗。
- get_window_obj: 计算窗口与标注的 IOF，筛选进入窗口的目标。
- crop_and_save_img / crop_img_withoutann: 生成 patch 及元信息。
- relocate: 将 patch 内预测框平移回整图。
- resize_bboxes_len6: 把不同尺度的框缩放回原图。
- merge_results_two_stage / merge_results_tensor: 多来源框统一旋转 NMS。


## 7. Global RoI 的重加权机制

global_roi_head 的 bbox_head 设置 use_reweight=True。

在 RotatedBBoxHead 中，对正样本回归权重 bbox_weights 进行了几何重加权：

- 基于 proposal 与 gt 的相对位移
- 结合 gt 长宽比与方向角
- 生成非均匀回归权重

该机制旨在增强对细长桥梁目标的回归约束。


## 8. 单阶段 ImgFPN 变体

对应：

- configs/bridge_benchmark/rotated_fcos_r50_fpn_2x_allbridge_ImgFPN_oc.py
- mmrotate/models/detectors/single_stage_ImgFPN.py

其核心思想与两阶段版本一致：

- 维持 local/global 双分支提取
- 推理阶段进行多尺度 global + local patch 联合
- 最后统一旋转 NMS

区别在于检测头为 RotatedFCOSHead，无 RPN/RoI 两级结构。


## 9. 与标准 MMRotate 模型的主要差异

相对标准 OrientedRCNN，本仓库 ImgFPN 两阶段模型新增：

1. 双分支参数化（local 与 global 各一套 backbone/neck/head）
2. 自定义数据流字段（g_img_list, g_img_infos）
3. 跨尺度特征裁剪与拼接融合（ConcateFPNFea）
4. 大图滑窗推理与跨来源结果聚合（local + multi-global）
5. global 分支独立损失与可选重加权回归头


## 10. 当前代码行为的注意点

在 LoadFPNImageFromFile.__call__ 中，read_global 在阈值判断后又被显式置为 False。

这意味着按当前代码默认路径：

- g_img_list/g_img_infos 通常为空
- 训练时 global 分支损失会进入“置零分支”
- 主要依赖 local 分支训练

但推理阶段 simple_test 仍会执行模型内部的金字塔 global-local 联合流程。

如需训练时真实启用 global 分支，需要恢复/修改 read_global 逻辑。


## 11. 一句话总结

该仓库的核心模型结构可以概括为：

“以 MMRotate 两阶段/单阶段检测器为骨架，叠加大幅面图像的多尺度全局-局部协同建模、特征融合与结果融合机制，用于桥梁目标的整体检测。”


## 12. 术语说明

- 含义：
	- “各自维护一套同构网络” 指 local 分支与 global 分支在结构上是相同的（同构），
		例如都是 ResNet-50 + FPN + 对应的 head，但它们是独立实例（参数不共享）。

- 代码位置（构建处）：
	- `mmrotate/models/detectors/two_stage_ImgFPN.py`：
		- `self.backbone = build_backbone(backbone)` 与 `self.global_backbone = build_backbone(backbone)`。
		- `self.neck` 与 `self.global_neck`。
		- `self.rpn_head` 与 `self.global_rpn_head`、`self.roi_head` 与 `self.global_roi_head`。
	- 数据管线：`mmrotate/datasets/pipelines/loading.py`（`LoadFPNImageFromFile` 输出 `g_img_list/g_img_infos`）。

- 设计目的：
	- local 负责高分辨率 patch 的细粒度定位；global 提供下采样后的全局语义与上下文。独立的网络允许各自学习专门化的特征表示，便于在 FPN 层或检测结果级别做更灵活的融合。

- 注意与代价：
	- 优点：表达更丰富、可独立调优局部与全局的能力。 
	- 缺点：计算与显存开销翻倍（两个完整网络）；训练需要数据管线提供 global 输入（当前 `LoadFPNImageFromFile` 有条件关闭逻辑）。
	- 折中：若想节省资源，可改为参数共享（同一 backbone 复用），但会丧失独立训练的灵活性。

