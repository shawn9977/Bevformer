import numpy as np
import torch
from torch.onnx import OperatorExportTypes
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint


@torch.no_grad()
def pytorch2onnx(
    config,
    checkpoint,
    output_file,
    opset_version=13,
    verbose=False,
    cuda=True,
    inputs_data=None,
):

    model = build_detector(config.model, test_cfg=config.get("test_cfg", None))
    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    if cuda:
        model.to("cuda")
    else:
        model.to("cpu")

    onnx_shapes = config.default_shapes
    input_shapes = config.input_shapes
    output_shapes = config.output_shapes
    dynamic_axes = config.dynamic_axes

    for key in onnx_shapes:
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        locals()[key] = onnx_shapes[key]

    torch.random.manual_seed(0)
    inputs = {}
    for key in input_shapes.keys():
        if inputs_data is not None and key in inputs_data:
            inputs[key] = inputs_data[key]
            if isinstance(inputs[key], np.ndarray):
                inputs[key] = torch.from_numpy(inputs[key])
            assert isinstance(inputs[key], torch.Tensor)
        else:
            for i in range(len(input_shapes[key])):
                if isinstance(input_shapes[key][i], str):
                    input_shapes[key][i] = eval(input_shapes[key][i])
            inputs[key] = torch.randn(*input_shapes[key])
        if cuda:
            inputs[key] = inputs[key].cuda()

    model.forward = model.forward_trt
    input_name = list(input_shapes.keys())
    output_name = list(output_shapes.keys())

    inputs = tuple(inputs.values())

    torch.onnx.export(
        model,
        inputs,
        output_file,
        input_names=input_name,
        output_names=output_name,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        verbose=verbose,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    )

    import openvino as ov

    # 创建新的 dummy inputs
    input_image = torch.randn(1, 6, 3, 480, 800)         # [1, 6, 3, 480, 800]
    prev_bev = torch.randn(2500, 1, 256)                 # [2500, 1, 256]
    use_prev_bev = torch.tensor([1])                     # [1]
    can_bus = torch.randn(18)                            # [18]
    lidar2img = torch.randn(1, 6, 4, 4)                  # [1, 6, 4, 4]

    # 创建 dummpy_inputs
    dummpy_inputs = (
        input_image,
        prev_bev,
        use_prev_bev,
        can_bus,
        lidar2img
    )

    # 手动跟踪模型，并转换为 OpenVINO IR 格式
    # traced_model = torch.jit.trace(
    # model,
    # example_inputs=dummpy_inputs,
    # strict=False,
    # check_trace=False,
    # )

    ov_model = ov.convert_model(model, example_input=dummpy_inputs)
    ov.save_model(ov_model, ov_dino_path)

    print(f"ONNX file has been saved in {output_file}")
