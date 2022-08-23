import time
from mmdet3d.apis import init_model, inference_detector, show_result_meshlab
import argparse
import glob

def parse_args():
    #currently not in use
    parser = argparse.ArgumentParser(description='Perform inference')
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', help='checkpoint or trained model file')
    parser.add_argument('--out', help='output folder')
    parser.add_argument('--point_cloud_path', help='point cloud path')
    args = parser.parse_args()
    
def calc_avg(curr_list):
    return sum(curr_list) / len(curr_list)

def main():

    # Debugging:
    # point_cloud = "/data/cmpe249-fa22/kitti/testing/velodyne/000001.bin"
    config_file = "../configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py"
    checkpoint_file = "/home/015306278/mmdetection3d2/mmdetection3d/work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/latest.pth"
    output_dir = "/home/015306278/mmdet3dInferenceResults"
    point_cloud_path = "/data/cmpe249-fa22/kitti/testing/velodyne"
    point_cloud_list= sorted(glob.glob(point_cloud_path+'/*.bin'))

    print("Point Cloud List Length:", len([point_cloud_list]))

    model = init_model(config_file, checkpoint_file, device='cpu')

    inference_times = []
    for i in range(300):
        pc = point_cloud_list[i]
        start_time = time.time()
        result, data = inference_detector(model, pc)
        finish_time = time.time()
    
        total_time = finish_time - start_time
        inference_times.append(total_time)
    
    avg_inference_time = calc_avg(inference_times)

    print("Average inference time: %f" %avg_inference_time)

    # Creates object files (prediction and point cloud) for visualization on inference
    show_result_meshlab(
        data,
        result,
        output_dir
    )

main()