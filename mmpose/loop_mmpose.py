import os
import time
import logging
import shutil
import json
import torch

from utilsMMpose import detection_inference, pose_inference

logging.basicConfig(level=logging.INFO)

logging.info("Waiting for data...")

def checkCudaPyTorch():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Found {num_gpus} GPU(s).")
    else:
        logging.info("No GPU detected. Exiting.")
        raise Exception("No GPU detected. Exiting.")

video_path = "/mmpose/data/video_mmpose.mov"
output_dir = "/mmpose/data/output_mmpose"

generateVideo=False

with open('/mmpose/defaultOpenCapSettings.json') as f:
    defaultOpenCapSettings = json.load(f)
model_config_person='/mmpose/faster_rcnn_r50_fpn_coco.py'
model_ckpt_person='/mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    
if os.path.isfile(video_path):
    os.remove(video_path)

checkCudaPyTorch()
while True:    
    if not os.path.isfile(video_path):
        time.sleep(0.1)
        continue

    logging.info("Processing mmpose...")
    # Re-read settings on each job so model_variant can change between requests.
    # In the mmpose container the shared Docker volume is mounted at /mmpose/data,
    # not /data (which is the mobilecap container's mount point).
    shared_settings_path = "/mmpose/data/defaultOpenCapSettings.json"
    if os.path.exists(shared_settings_path):
        with open(shared_settings_path) as _sf:
            _shared = json.load(_sf)
        model_type = _shared.get('active_pose_model', 'hrnet')
    else:
        model_type = 'hrnet'

    if model_type == 'vitpose':
        # Register the ViT backbone with mmpose before build_posenet is called.
        # The base Docker image (mmpose ~v0.13) predates ViTPose and does not
        # include this backbone; importing the local file registers it.
        import vit_backbone  # noqa: F401
        model_config_pose = '/mmpose/vitpose_base_coco_wholebody.py'
        model_ckpt_pose   = '/mmpose/vitpose-b-wholebody.pth'
        bbox_thr = defaultOpenCapSettings.get('vitpose', 0.8)
        logging.info("Using ViTPose model.")
    else:
        model_config_pose = '/mmpose/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
        model_ckpt_pose   = '/mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
        bbox_thr = defaultOpenCapSettings.get('hrnet', 0.8)
        logging.info("Using HRNet model.")

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    try:
        checkCudaPyTorch()
        # Run human detection.
        pathModelCkptPerson = model_ckpt_person
        bboxPath = os.path.join(output_dir, 'box.pkl')
        full_model_config_person = model_config_person
        detection_inference(full_model_config_person, pathModelCkptPerson,
                            video_path, bboxPath)        
        
        # Run pose detection.     
        pathModelCkptPose = model_ckpt_pose
        pklPath = os.path.join(output_dir, 'human.pkl')
        videoOutPath = ''
        full_model_config_pose = model_config_pose
        pose_inference(full_model_config_pose, pathModelCkptPose, 
                       video_path, bboxPath, pklPath, videoOutPath, 
                       bbox_thr=bbox_thr, visualize=generateVideo)
        if os.path.isfile(video_path):
            os.remove(video_path)
        if os.path.isfile(bboxPath):
            os.remove(bboxPath)
        
        logging.info("mmpose: Done. Cleaning up")
        
    except:
        logging.info("mmpose: Pose detection failed.")
        logging.info("mmpose: Exception: %s", traceback.format_exc())
        os.remove(video_path)
