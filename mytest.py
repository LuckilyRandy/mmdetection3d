from mmdet3d.apis import init_model, inference_detector
from mmdet3d.registry import VISUALIZERS

"""
# pointpillars
configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py
checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth

# second
configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py
checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth

# fcos3d
configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py
checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth


"""

config_file = 'configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py'
checkpoint_file = 'checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth'
model = init_model(config_file, checkpoint_file)
print("=======model初始化成功======")
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
print("=======visualizer初始化成功======")
result, data = inference_detector(model, 'demo/data/kitti/000008.bin')
print("=======推理结束======")
points = data['inputs']['points']
data_input = dict(points=points)
# show
visualizer.add_datasample(
    'result',
    data_input,
    data_sample=result,
    draw_gt=False,
    show=True,
    wait_time=-1,
    out_file="./out",
    pred_score_thr=0.0,
    vis_task='lidar_det')

