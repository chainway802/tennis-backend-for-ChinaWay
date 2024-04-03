# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/03 18:04
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from abc import ABC

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from .AbstractModel import AbstractModel


class TRTModel(AbstractModel, ABC):
    def __init__(self, max_batch_size=1):
        self._output_names = None
        self.output_dims = {}
        self.max_batch_size = max_batch_size

    def init_model(self, engine_path):
        self.engine_path = engine_path
        self.engine = self._load_engine(self.engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def inference(self, x: np.ndarray):
        input_name = list(self.inputs.keys())[0]  # 只针对单输入
        dtype = self._trt_to_np_dtype(self.engine.get_tensor_dtype(input_name))
        x = x.astype(dtype)
        np.copyto(self.inputs[input_name].host, x.ravel())

        for inp in self.inputs.values():
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs.values():
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        outputs = {}
        for key, out in self.outputs.items():
            outputs[key] = out.host.reshape((self.max_batch_size, *self.output_dims[key]))

        return outputs

    def _load_engine(self, engine_path) -> trt.ICudaEngine:
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, mode='rb') as f:
                engine_bytes = f.read()
            trt.init_libnvinfer_plugins(logger, namespace='')
        return runtime.deserialize_cuda_engine(engine_bytes)

    def _allocate_buffers(self):
        inputs = {}
        outputs = {}
        bindings = []
        stream = cuda.Stream()

        for idx, binding in enumerate(self.engine):
            dims = self.engine.get_tensor_shape(binding)[1:]
            if dims[-1] == -1:
                dims = (3, 320, 320)
            size = trt.volume(dims) * self.max_batch_size
            dtype = self._trt_to_np_dtype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(binding, (self.max_batch_size, *dims))
                inputs[binding] = HostDeviceMem(host_mem, device_mem)
            else:
                outputs[binding] = HostDeviceMem(host_mem, device_mem)
                self.output_dims[binding] = dims

        return inputs, outputs, bindings, stream

    def _trt_to_np_dtype(trt_dtype):
        """Convert TensorRT dtype to NumPy dtype."""
        if trt_dtype == trt.DataType.BOOL:
            return np.bool_  # 使用 np.bool_ 而不是 np.bool
        elif trt_dtype == trt.DataType.INT8:
            return np.int8
        elif trt_dtype == trt.DataType.INT32:
            return np.int32
        elif trt_dtype == trt.DataType.FLOAT:
            return np.float32
        elif trt_dtype == trt.DataType.HALF:
            return np.float16
        # 添加其他必要的数据类型转换
        else:
            raise TypeError(f'Unsupported TensorRT data type: {trt_dtype}')

    def release(self):
        # 释放CUDA内存
        for input_host_device in self.inputs.values():
            input_host_device.device.free()
        for output_host_device in self.outputs.values():
            output_host_device.device.free()

        # 释放TensorRT资源
        del self.context
        del self.engine
        # 如果有其他需要释放的资源，也应在这里处理


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
