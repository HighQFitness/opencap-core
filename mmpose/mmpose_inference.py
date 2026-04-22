import copy
import mmcv
import numpy as np
import torch

from mmpose_utils import LoadImage
from mmcv.runner import load_checkpoint
from mmpose.apis import get_track_id
from mmpose.models import build_posenet
from mmpose.datasets.pipelines import Compose

# Register ViTPose++ (ViTMoE + TopDownMoE) before build_posenet.
import vit_moe  # noqa: F401
import top_down_moe  # noqa: F401

# Associate head index for COCO-WholeBody (133 kpts); training dataset_idx=5.
_VITPOSEPP_WHOLEBODY_HEAD_IDX = 4
_VITPOSEPP_WHOLEBODY_DATASET_IDX = 5


def uses_vitpose_plusplus_moe(model):
    """True if model is ViTPose++ TopDownMoE with a whole-body associate head."""
    cfg = getattr(model, 'cfg', None)
    if cfg is None:
        return False
    if getattr(cfg.model, 'type', None) != 'TopDownMoE':
        return False
    heads = getattr(model, 'associate_keypoint_heads', None)
    return heads is not None and len(heads) > _VITPOSEPP_WHOLEBODY_HEAD_IDX

def init_pose_model(config, checkpoint, device="cuda:0"):
    """Initialize pose model from config file and checkpoint path

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path.
    Returns:
        nn.Module: The constructed detector.
    """
    config = mmcv.Config.fromfile(config)
    config.model.pretrained = None

    model = build_posenet(config.model)
    load_checkpoint(model, checkpoint, map_location=device)
    model.cfg = config
    model.to(device).eval()

    return model


def init_test_pipeline(model):
    """Initialize testing pipeline

    Args:
        model (nn.Module): inference model with config attribute
    Returns:
        pipeline (list[dict | callable]): A sequence of data transforms
    """
    channel_order = model.cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + model.cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    return test_pipeline


def run_pose_inference(model, batch, save_features=False, save_heatmap=False):
    """Defines computations performed for pose inference.

    Args:
        model (nn.Module): inference model with config attribute
        batch (dict): data dictionary with img and img_metas key
        save_feautres (bool): save feature maps
        save_heatmap (bool): save keypoint heatmaps
    Returns:
        result (dict): result dictionary with saved tensors
    """
    img, img_metas = batch['img'], batch['img_metas']
    assert img.size(0) == len(img_metas)
    batch_size, _, img_height, img_width = img.shape

    result = {}
    features = model.backbone(img)
    if model.with_neck:
        features = model.neck(features)
    output_heatmap = model.keypoint_head.inference_model(
                            features, flip_pairs=None)
    keypoint_result = model.keypoint_head.decode(
        img_metas, output_heatmap, img_size=[img_width, img_height])
    if save_features:
        if type(features) is list:
            if type(features[0]) is list:
                features = sum(features, [])
            result['features'] = [x.cpu().numpy() for x in features]
        else:
            result['features'] = features.cpu().numpy()
    if save_heatmap:
        result['output_heatmap'] = output_heatmap
    result['preds'] = keypoint_result['preds']
    result['bbox'] = np.stack([x['image_file'] for x in img_metas])

    img_flipped = img.flip(3)
    features_flipped = model.backbone(img_flipped)
    if model.with_neck:
        features_flipped = model.neck(features_flipped)
    output_flipped_heatmap = model.keypoint_head.inference_model(
        features_flipped, img_metas[0]['flip_pairs'])
    output_heatmap_flipped_avg = (output_heatmap +
                                  output_flipped_heatmap) * 0.5
    keypoint_with_flip_result = model.keypoint_head.decode(
        img_metas, output_heatmap_flipped_avg, img_size=[img_width, img_height])
    if save_features:
        if type(features_flipped) is list:
            if type(features_flipped[0]) is list:
                features_flipped = sum(features_flipped, [])
            result['features_flipped'] = [x.cpu().numpy() for x in features_flipped]
        else:
            result['features_flipped'] = features_flipped.cpu().numpy()
    if save_heatmap:
        result['output_heatmap_flipped_avg'] = output_heatmap_flipped_avg
    result['preds_with_flip'] = keypoint_with_flip_result['preds']

    return result


def run_pose_inference_moe_wholebody(model, batch, save_features=False,
                                   save_heatmap=False):
    """ViTPose++: run the COCO-WholeBody associate head (not the 17-kpt COCO head)."""
    img, img_metas = batch['img'], batch['img_metas']
    assert img.size(0) == len(img_metas)
    batch_size, _, img_height, img_width = img.shape
    device = img.device

    for m in img_metas:
        m['dataset_idx'] = _VITPOSEPP_WHOLEBODY_DATASET_IDX

    img_sources = torch.full(
        (batch_size,), _VITPOSEPP_WHOLEBODY_DATASET_IDX,
        dtype=torch.long, device=device)

    result = {}
    features = model.backbone(img, img_sources)
    if model.with_neck:
        features = model.neck(features)

    head = model.associate_keypoint_heads[_VITPOSEPP_WHOLEBODY_HEAD_IDX]
    output_heatmap = head.inference_model(features, flip_pairs=None)
    keypoint_result = head.decode(
        img_metas, output_heatmap, img_size=[img_width, img_height])

    if save_features:
        if isinstance(features, list):
            if isinstance(features[0], list):
                features = sum(features, [])
            result['features'] = [x.cpu().numpy() for x in features]
        else:
            result['features'] = features.cpu().numpy()
    if save_heatmap:
        result['output_heatmap'] = output_heatmap
    result['preds'] = keypoint_result['preds']
    result['bbox'] = np.stack([x['image_file'] for x in img_metas])

    if model.test_cfg.get('flip_test', True):
        img_flipped = img.flip(3)
        features_flipped = model.backbone(img_flipped, img_sources)
        if model.with_neck:
            features_flipped = model.neck(features_flipped)
        output_flipped_heatmap = head.inference_model(
            features_flipped, img_metas[0]['flip_pairs'])
        output_heatmap_flipped_avg = (output_heatmap + output_flipped_heatmap) * 0.5
        keypoint_with_flip_result = head.decode(
            img_metas, output_heatmap_flipped_avg,
            img_size=[img_width, img_height])
        if save_features:
            if isinstance(features_flipped, list):
                if isinstance(features_flipped[0], list):
                    features_flipped = sum(features_flipped, [])
                result['features_flipped'] = [
                    x.cpu().numpy() for x in features_flipped]
            else:
                result['features_flipped'] = features_flipped.cpu().numpy()
        if save_heatmap:
            result['output_heatmap_flipped_avg'] = output_heatmap_flipped_avg
        result['preds_with_flip'] = keypoint_with_flip_result['preds']
    else:
        result['preds_with_flip'] = keypoint_result['preds']

    return result


def run_pose_tracking(results):
    next_id = 0
    pose_result_last = []
    pose_tracked_results = []
    for pose_result in results:
        for instance in pose_result:
            instance['keypoints'] = instance['preds_with_flip']
        pose_result, next_id = get_track_id(pose_result, pose_result_last, next_id,
                                           use_oks=False, tracking_thr=0.3)
        pose_result_last = copy.deepcopy(pose_result)
        for instance in pose_result:
            del instance['keypoints']
        pose_tracked_results.append(pose_result)
    return pose_tracked_results
