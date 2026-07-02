"""
Inference backend wrappers for evaluating exported ONNX and TensorRT models.

Suggested usage in model_test.py:

    from inference_backends import (
        infer_runtime_from_model_path,
        ONNXRuntimeModel,
        TensorRTEngineModel,
    )

    runtime = params_dict.get("runtime") or infer_runtime_from_model_path(model_path)

    if runtime == "onnx":
        model = ONNXRuntimeModel(model_path)
    elif runtime == "tensorrt":
        model = TensorRTEngineModel(model_path, device=device)

These wrappers are callable and expose eval(), so they can be used in places
where model_test.py currently expects a PyTorch-like model object.
"""

import os
from typing import Optional, Sequence

import numpy as np
import torch


class ONNXRuntimeModel:
    """
    ONNX Runtime wrapper with PyTorch-like callable interface.

    Important:
    - Inputs are matched to ONNX input names by order.
    - Outputs are returned as torch tensors.
    - By default, outputs stay on CPU, because ONNX Runtime returns NumPy arrays.
    """

    def __init__(
        self,
        model_path,
        providers=None,
        device="cpu",
        debug=False,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX evaluation requires onnxruntime or onnxruntime-gpu."
            ) from exc

        self.model_path = model_path
        self.device = torch.device(device)
        self.debug = debug

        if providers is None:
            available = ort.get_available_providers()

            # For debugging, prefer CPU first because your notebook comparison used CPU.
            preferred = [
                "CPUExecutionProvider",
            ]

            providers = [p for p in preferred if p in available]

            if not providers:
                providers = available

        self.session = ort.InferenceSession(
            model_path,
            providers=list(providers),
        )

        self.input_metas = self.session.get_inputs()
        self.output_metas = self.session.get_outputs()

        self.input_names = [inp.name for inp in self.input_metas]
        self.output_names = [out.name for out in self.output_metas]

        print("Loaded ONNX model:", model_path)
        print("ONNX providers:", self.session.get_providers())

        if self.debug:
            print("=== ONNX inputs ===")
            for i, inp in enumerate(self.input_metas):
                print(i, inp.name, inp.shape, inp.type)

            print("=== ONNX outputs ===")
            for i, out in enumerate(self.output_metas):
                print(i, out.name, out.shape, out.type)

    def eval(self):
        return self

    def __call__(self, *inputs):
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"ONNX model expects {len(self.input_names)} input(s), "
                f"but received {len(inputs)}. "
                f"ONNX input names are: {self.input_names}"
            )

        feed_dict = {}

        for input_name, tensor in zip(self.input_names, inputs):
            if torch.is_tensor(tensor):
                array = tensor.detach().cpu().numpy()
            else:
                array = np.asarray(tensor)

            # For model inputs, force float32. Your debug showed x is float32.
            if np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32, copy=False)

            feed_dict[input_name] = np.ascontiguousarray(array)

            if self.debug:
                print(
                    f"ONNX feed {input_name}:",
                    "shape=", feed_dict[input_name].shape,
                    "dtype=", feed_dict[input_name].dtype,
                    "min=", feed_dict[input_name].min(),
                    "max=", feed_dict[input_name].max(),
                    "mean=", feed_dict[input_name].mean(),
                )

        # Request outputs by explicit ONNX output names.
        outputs = self.session.run(self.output_names, feed_dict)

        torch_outputs = tuple(
            torch.from_numpy(np.ascontiguousarray(output))
            for output in outputs
        )

        if self.debug:
            for name, out in zip(self.output_names, torch_outputs):
                print(
                    f"ONNX output {name}:",
                    "shape=", tuple(out.shape),
                    "dtype=", out.dtype,
                    "device=", out.device,
                    "min=", out.min().item(),
                    "max=", out.max().item(),
                    "mean=", out.float().mean().item(),
                )

        return torch_outputs[0] if len(torch_outputs) == 1 else torch_outputs

def _torch_dtype_from_trt_dtype(trt_dtype):
    """
    Convert TensorRT dtype to torch dtype.
    """
    import tensorrt as trt

    dtype_map = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }

    if hasattr(trt, "int64"):
        dtype_map[trt.int64] = torch.int64

    try:
        return dtype_map[trt_dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}") from exc


class TensorRTEngineModel:
    """
    TensorRT serialized engine wrapper with a PyTorch-like callable interface.

    Expected input model file extensions:
        .engine
        .plan

    This implementation targets TensorRT 10.x style I/O tensor APIs:
        engine.num_io_tensors
        engine.get_tensor_name()
        engine.get_tensor_mode()
        context.set_input_shape()
        context.set_tensor_address()
        context.execute_async_v3()

    Inputs and outputs are kept as PyTorch CUDA tensors.
    """

    def __init__(self, engine_path: str, device) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT evaluation requires a CUDA-capable GPU.")

        self.device = torch.device(device)

        if self.device.type != "cuda":
            raise ValueError(
                "TensorRT evaluation requires a CUDA device, "
                f"but got device={device!r}."
            )

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "TensorRT evaluation requires the `tensorrt` Python package."
            ) from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as engine_file:
            serialized_engine = engine_file.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context.")

        self.input_names = []
        self.output_names = []

        for tensor_index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(tensor_index)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)

            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
            else:
                self.output_names.append(tensor_name)

    def eval(self) -> "TensorRTEngineModel":
        return self

    def __call__(self, *inputs):
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"TensorRT engine expects {len(self.input_names)} input(s), "
                f"but received {len(inputs)}."
            )

        retained_inputs = []

        for tensor_name, tensor in zip(self.input_names, inputs):
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor, device=self.device)
            else:
                tensor = tensor.to(self.device)

            tensor = tensor.contiguous()
            retained_inputs.append(tensor)

            if not self.context.set_input_shape(tensor_name, tuple(tensor.shape)):
                raise RuntimeError(
                    f"TensorRT rejected dynamic input shape {tuple(tensor.shape)} "
                    f"for tensor {tensor_name!r}."
                )

            self.context.set_tensor_address(tensor_name, tensor.data_ptr())

        outputs = []

        for tensor_name in self.output_names:
            output_shape = tuple(self.context.get_tensor_shape(tensor_name))

            if any(dim < 0 for dim in output_shape):
                raise RuntimeError(
                    f"TensorRT output shape for {tensor_name!r} is unresolved: "
                    f"{output_shape}"
                )

            output_dtype = _torch_dtype_from_trt_dtype(
                self.engine.get_tensor_dtype(tensor_name)
            )

            output = torch.empty(
                output_shape,
                dtype=output_dtype,
                device=self.device,
            )

            self.context.set_tensor_address(tensor_name, output.data_ptr())
            outputs.append(output)

        stream = torch.cuda.current_stream(device=self.device)

        succeeded = self.context.execute_async_v3(stream.cuda_stream)

        if not succeeded:
            raise RuntimeError("TensorRT execute_async_v3() returned False.")

        stream.synchronize()

        return outputs[0] if len(outputs) == 1 else tuple(outputs)
