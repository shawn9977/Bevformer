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


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("ir_model", help="OpenVINO IR model (.xml file)")
    args = parser.parse_args()
    return args


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
            input_blobs["image"].name: img.reshape(-1).astype(np.float32),
            input_blobs["prev_bev"].name: prev_bev.reshape(-1).astype(np.float32),
            input_blobs["use_prev_bev"].name: use_prev_bev.reshape(-1).astype(np.float32),
            input_blobs["can_bus"].name: can_bus.reshape(-1).astype(np.float32),
            input_blobs["lidar2img"].name: lidar2img.reshape(-1).astype(np.float32)
        }

        # Perform inference
        start_time = time.time()
        infer_request.infer(inputs=inputs)
        end_time = time.time()

        # Extract outputs
        output_classes = infer_request.outputs[output_blob_names[0]]  # Adjust this based on your output blob name
        bev_embed = output_classes  # Adjust as per your model output structure

        # Post-process results
        results.extend(
            post_process(output_classes, img_metas)  # Define post_process function accordingly
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
