
source activate mmrot


python tools/test.py --config configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py\
                     --checkpoint xxx\
                     --format-only \
                     --eval-options submission_dir=work_dirs/bridge_results
