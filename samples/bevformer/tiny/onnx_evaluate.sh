#!/bin/bash

gpu_id=0
while getopts "d:" opt
do
  case $opt in
    d)
      gpu_id=$OPTARG
      ;;
    ?)
      echo "There is unrecognized parameter."
      exit 1
      ;;
  esac
done

echo "Running on the GPU: $gpu_id"

CUDA_VISIBLE_DEVICES=$gpu_id python tools/bevformer/evaluate_onnx.py \
configs/bevformer/bevformer_tiny_trt.py \
checkpoints/onnx/bevformer_tiny_epoch_24.onnx
