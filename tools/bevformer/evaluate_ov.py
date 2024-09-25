from openvino.runtime import Core
import numpy as np
import copy
import time
import mmcv
import argparse
import sys
from mmcv import Config
from mmcv.runner import load_checkpoint
import torch
from mmcv.parallel import MMDataParallel

sys.path.append(".")
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


from mmdet.models import DETECTORS
from third_party.bev_mmdet3d import BEVFormer
from third_party.bev_mmdet3d.core.bbox import bbox3d2result


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("ir_model", help="OpenVINO IR model (.xml file)")
    args = parser.parse_args()
    return args


def denormalize_bbox(normalized_bboxes, pc_range=None):
    # rotation
    # 提取旋转角度的正弦值
    rot_sine = normalized_bboxes[..., 6:7]
    
    # 提取旋转角度的余弦值
    rot_cosine = normalized_bboxes[..., 7:8]
    
    # 使用反正切函数计算旋转角度（弧度制）
    # atan2 可以根据正弦和余弦值正确确定角度的象限
    rot = np.arctan2(rot_sine, rot_cosine)
    
    # center in the bev (Bird's Eye View)
    # 提取边界框中心点的 x 坐标
    cx = normalized_bboxes[..., 0:1]
    
    # 提取边界框中心点的 y 坐标
    cy = normalized_bboxes[..., 1:2]
    
    # 提取边界框中心点的 z 坐标
    # 注意这里 z 坐标的索引与 x, y 不同，可能是数据格式特有的排列方式
    cz = normalized_bboxes[..., 4:5]
    
    # size
    # 提取边界框的宽度
    w = normalized_bboxes[..., 2:3]
    
    # 提取边界框的长度
    l = normalized_bboxes[..., 3:4]
    
    # 提取边界框的高度
    h = normalized_bboxes[..., 5:6]
    
    # 对宽度、长度和高度进行指数运算，通常是因为它们在网络中是以对数形式输出的
    w = np.exp(w)
    l = np.exp(l)
    h = np.exp(h)
    
    # 检查是否有额外的速度信息
    if normalized_bboxes.shape[-1] > 8:
        # velocity
        # 提取速度的 x 方向分量
        vx = normalized_bboxes[:, 8:9]
        
        # 提取速度的 y 方向分量
        vy = normalized_bboxes[:, 9:10]
        
        # 将所有信息（中心点坐标、尺寸、旋转角度、速度）拼接起来
        # axis=-1 表示沿着最后一个维度拼接
        denormalized_bboxes = np.concatenate([cx, cy, cz, w, l, h, rot, vx, vy], axis=-1)
    else:
        # 如果没有速度信息，只拼接中心点坐标、尺寸和旋转角度
        denormalized_bboxes = np.concatenate([cx, cy, cz, w, l, h, rot], axis=-1)
    
    # 返回拼接后的边界框信息
    return denormalized_bboxes


# 定义常量
MAX_NUM = 300  # 最大选取的框数量
NUM_CLASSES = 10  # 类别数量
POST_CENTER_RANGE = np.array([-61.2, -61.2, -10.0, 61.2, 61.2, 10.0])
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

def sigmoid(x):
    """计算 Sigmoid 函数"""
    return 1 / (1 + np.exp(-x))

def decode_single(cls_scores, bbox_preds):
    """Decode bboxes.
    Args:
        cls_scores (Tensor): Outputs from the classification head, \
            shape [num_query, cls_out_channels]. Note \
            cls_out_channels should includes background.
        bbox_preds (Tensor): Outputs from the regression \
            head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
            Shape [num_query, 9].
    Returns:
        list[dict]: Decoded boxes.
    """
    max_num = MAX_NUM  # self.max_num  

    cls_scores = sigmoid(cls_scores)

    # 使用 NumPy 处理最大值和索引
    flat_scores = cls_scores.flatten()
    if len(flat_scores) > max_num:
        indexs = np.argpartition(flat_scores, -max_num)[-max_num:]  # 获取前 max_num 的索引
        scores = flat_scores[indexs]  # 获取对应的分数
        # 排序
        sorted_indices = np.argsort(scores)[::-1]  # 降序排序
        scores = scores[sorted_indices]
        indexs = indexs[sorted_indices]
    else:
        scores = flat_scores
        indexs = np.arange(len(flat_scores))

    
    labels = indexs % NUM_CLASSES    # self.num_classes
    bbox_index = indexs // NUM_CLASSES  # 用 NumPy 进行整除
    bbox_preds = bbox_preds[bbox_index]

    final_box_preds = denormalize_bbox(bbox_preds, PC_RANGE)
    final_scores = scores
    final_preds = labels

    mask = (final_box_preds[..., :3] >= POST_CENTER_RANGE[:3]).all(1)
    mask &= (final_box_preds[..., :3] <= POST_CENTER_RANGE[3:]).all(1)

    boxes3d = final_box_preds[mask]
    scores = final_scores[mask]

    labels = final_preds[mask]
    predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}

    return predictions_dict




def decode(preds_dicts):
    """Decode bboxes.
    Args:
        all_cls_scores (Tensor): Outputs from the classification head, \
            shape [nb_dec, bs, num_query, cls_out_channels]. Note \
            cls_out_channels should includes background.
        all_bbox_preds (Tensor): Sigmoid outputs from the regression \
            head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
            Shape [nb_dec, bs, num_query, 9].
    Returns:
        list[dict]: Decoded boxes.
    """
    all_cls_scores = preds_dicts["all_cls_scores"][-1]
    all_bbox_preds = preds_dicts["all_bbox_preds"][-1]

    # 检查类型
    if isinstance(all_cls_scores, np.ndarray):
        batch_size = all_cls_scores.shape[0]  # NumPy 数组
    else:
        batch_size = all_cls_scores.size(0)  # PyTorch 张量

    predictions_list = []
    for i in range(batch_size):
        predictions_list.append(
            decode_single(all_cls_scores[i], all_bbox_preds[i])
        )
    return predictions_list



def get_bboxes(preds_dicts, img_metas, rescale=False):
    """Generate bboxes from bbox head predictions.
    Args:
        preds_dicts (tuple[list[dict]]): Prediction results.
        img_metas (list[dict]): Point cloud and image's meta info.
    Returns:
        list[dict]: Decoded bbox, scores and labels after nms.
    """

    preds_dicts = decode(preds_dicts)

    num_samples = len(preds_dicts)
    ret_list = []
    for i in range(num_samples):
        preds = preds_dicts[i]
        bboxes = preds["bboxes"]

        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

        code_size = bboxes.shape[-1]
        bboxes = img_metas[i]["box_type_3d"](bboxes, code_size)
        scores = preds["scores"]
        labels = preds["labels"]

        ret_list.append([bboxes, scores, labels])

    return ret_list


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (ndarray): Bounding boxes with shape of (n, 5).
        labels (ndarray): Labels with shape of (n, ).
        scores (ndarray): Scores with shape of (n, ).
        attrs (ndarray, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, ndarray]: Bounding box results in numpy mode.

            - boxes_3d (ndarray): 3D boxes.
            - scores (ndarray): Prediction scores.
            - labels_3d (ndarray): Box labels.
            - attrs_3d (ndarray, optional): Box attributes.
    """
    result_dict = {
        "boxes_3d": bboxes,  # 不需要转换
        "scores_3d": scores,
        "labels_3d": labels
    }

    if attrs is not None:
        result_dict["attrs_3d"] = attrs

    return result_dict

def post_process(outputs_classes, outputs_coords, img_metas):
    dic = {"all_cls_scores": outputs_classes, "all_bbox_preds": outputs_coords}
    result_list = get_bboxes(dic, img_metas, rescale=True)

    return [
        {
            "pts_bbox": bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in result_list
        }
    ]


def main():
    args = parse_args()

    config_file = args.config
    ir_model_file = args.ir_model
    config = Config.fromfile(config_file)

    if hasattr(config, "plugin"):
        import importlib
        import sys

        sys.path.append(".")
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    dataset = build_dataset(cfg=config.data.val)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    # Initialize OpenVINO Runtime (Inference Engine)
    ie = Core()
    model = ie.read_model(model=ir_model_file)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    infer_request = compiled_model.create_infer_request()

    # Get input and output blob names
    input_blobs = compiled_model.inputs
    output_blobs = compiled_model.outputs

    print(type(input_blobs))
    print(type(output_blobs))


    # 假设你已经有了以下变量
    img = ...  # 图像数据
    prev_bev = ...  # 前一帧的 BEV 数据
    use_prev_bev = ...  # 是否使用前一帧 BEV 的标志
    can_bus = ...  # CAN 总线数据
    lidar2img = ...  # LiDAR 到图像的变换矩阵

    # 创建一个字典来存储输入名称和对应的 Blob 对象
    input_names_to_blobs = {}

    # 遍历所有输入 Blob，提取名称和形状
    for blob in input_blobs:
        name = blob.any_name  # 获取输入名称
        shape = blob.shape  # 获取输入形状
        input_names_to_blobs[name] = blob


    # Initialize variables
    ts = []
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    prev_bev = np.random.randn(config.bev_h_ * config.bev_w_, 1, config._dim_)
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }

    # Inference loop
    for data in loader:
        img = data["img"][0].data[0].numpy()
        img_metas = data["img_metas"][0].data[0]

        use_prev_bev = np.array([1.0])
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = np.array([0.0])
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        can_bus = img_metas[0]["can_bus"].astype(np.float32)
        lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0).astype(np.float32)

        # Prepare inputs as a dictionary with correct input names
        inputs = {
            input_names_to_blobs["image"].any_name: img.reshape(input_names_to_blobs["image"].shape).astype(np.float32),
            input_names_to_blobs["prev_bev"].any_name: prev_bev.reshape(input_names_to_blobs["prev_bev"].shape).astype(np.float32),
            input_names_to_blobs["use_prev_bev"].any_name: use_prev_bev.reshape(input_names_to_blobs["use_prev_bev"].shape).astype(np.float32),
            input_names_to_blobs["can_bus"].any_name: can_bus.reshape(input_names_to_blobs["can_bus"].shape).astype(np.float32),
            input_names_to_blobs["lidar2img"].any_name: lidar2img.reshape(input_names_to_blobs["lidar2img"].shape).astype(np.float32)
        }

        # Perform inference
        start_time = time.time()
        infer_request.infer(inputs=inputs)
        end_time = time.time()

        # Extract outputs
        #output_classes = infer_request.outputs[output_blob_names[0]]  # Adjust this based on your output blob name
        #bev_embed = output_classes  # Adjust as per your model output structure


        # 获取输出名称
        output_names = [blob.any_name for blob in output_blobs]

        # 获取输出张量并读取数据
        output_data = []
        for name in output_names:
            output_tensor = infer_request.get_output_tensor(output_names.index(name))
            output_data.append(output_tensor.data)

        # 输出结果
        bev_embed = output_data[0]
        output_classes = output_data[1]
        outputs_coords = output_data[2]
        # 打印结果
        print("bev_embed shape:", bev_embed.shape)
        print("output_classes shape:", output_classes.shape)
        print("outputs_coords shape:", outputs_coords.shape)

        # Post-process results
        results.extend(
            post_process(output_classes, outputs_coords, img_metas)
        )

        prev_bev = bev_embed
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        ts.append(end_time - start_time)

        for _ in range(len(img)):
            prog_bar.update()

    # Evaluate metrics
    metric = dataset.evaluate(results)    

    # Summary
    print("*" * 50 + " SUMMARY " + "*" * 50)
    for key in metric.keys():
        if key == "pts_bbox_NuScenes/NDS":
            print(f"NDS: {round(metric[key], 3)}")
        elif key == "pts_bbox_NuScenes/mAP":
            print(f"mAP: {round(metric[key], 3)}")

    latency = round(sum(ts[1:-1]) / len(ts[1:-1]) * 1000, 2)
    print(f"Latency: {latency}ms")
    print(f"FPS: {1000 / latency}")


if __name__ == "__main__":
    main()
