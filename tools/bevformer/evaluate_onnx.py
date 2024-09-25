import onnxruntime as ort
import numpy as np
import torch
import copy
from mmcv import Config
import argparse
import sys

sys.path.append(".")
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset



def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    config_file = args.config
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

    
    # Load the ONNX model
    checkpoint_file = args.checkpoint
    session = ort.InferenceSession(checkpoint_file)

    
    # Build the dataset and dataloader
    dataset = build_dataset(cfg=config.data.val)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    # Prepare for inference
    prev_bev = np.random.randn(config.bev_h_ * config.bev_w_, 1, config._dim_).astype(np.float32)
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }

    # Inference loop
    ts = []
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for data in loader:
        img = data["img"][0].data[0].numpy().astype(np.float32)
        img_metas = data["img_metas"][0].data[0]

        use_prev_bev = np.array([1.0], dtype=np.float32)
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = np.array([0.0], dtype=np.float32)
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        can_bus = np.array(img_metas[0]["can_bus"], dtype=np.float32)
        lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0).astype(np.float32)

        # Prepare inputs
        inputs = {
            "image": img,
            "prev_bev": prev_bev,
            "use_prev_bev": use_prev_bev,
            "can_bus": can_bus,
            "lidar2img": lidar2img
        }

        # Run inference
        input_names = [i.name for i in session.get_inputs()]
        output_names = [o.name for o in session.get_outputs()]
        ort_inputs = {name: inputs[name] for name in input_names}
        
        import time
        start_time = time.time()
        ort_outputs = session.run(output_names, ort_inputs)
        end_time = time.time()

        # Process outputs
        bev_embed = ort_outputs[0]
        outputs_classes = ort_outputs[1]
        outputs_coords = ort_outputs[2]

        results.extend(
            pth_model.post_process(outputs_classes, outputs_coords, img_metas)
        )

        prev_bev = bev_embed
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        ts.append(end_time - start_time)

        for _ in range(len(img)):
            prog_bar.update()

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
