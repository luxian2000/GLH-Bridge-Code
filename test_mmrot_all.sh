
source activate mmrot

# 1.Please refer to https://github.com/open-mmlab/mmrotate/blob/b030f38909fc431be7ecb90772ac30da9da29bcb/docs/zh_cn/get_started.md?plain=1#L26 to prepare your result
# 2.Then submit the result to the benchmark website: https://www.codabench.org/competitions/3371
python tools/test.py --config configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py\
                     --checkpoint xxx\
                     --format-only \
                     --eval-options submission_dir=work_dirs/bridge_results
