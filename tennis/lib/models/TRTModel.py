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
import onnxruntime as ort
from .AbstractModel import AbstractModel
import os

class TRTModel(AbstractModel, ABC):
    def __init__(self, 
                 engine_file_path=None,
                 onnx_file_path=None,
                 use_onnx=False,
                 precision_flop="FP16",
                 img_size=(640, 640),
                 dynamic_shapes={},
                 dynamic_max_batch_size=1):
        
        # self.max_batch_size = max_batch_size
        self.cfx = cuda.Device(0).make_context()
        self.engine_file_path = engine_file_path
        self.onnx_file_path = onnx_file_path
        self.use_onnx = use_onnx
        self.precision_flop = precision_flop
        self.img_size = img_size
        self.dynamic_shapes = dynamic_shapes
        self.dynamic_max_batch_size = dynamic_max_batch_size
        self.inputs = []
        self.inputs_1 = {}
        self.outputs = []
        self.outputs_1 = {}
        self.bindings = []
        self.output_dims = {}
        # load_tensorrt_plugin() 无第三方算子库，暂且注释掉

    def init_model(self):
        """
        加载ONNX模型 
        """
        if self.use_onnx:
            # 初始化onnx模型
            self.ort_session = ort.InferenceSession(self.onnx_file_path, providers=['CUDAExecutionProvider'])
            return
        else:
            self._load_engine()
            self.context = self.engine.create_execution_context()
            assert self.context, "创建的上下文管理器 context 为空，请检查相应的操作"

            self._allocate_buffers()
            self.inputshape = self.engine.get_tensor_shape(list(self.inputs_1.keys())[0])[-2:]
        
    def inference(self, x: np.ndarray):
        pass
        # input_name = list(self.inputs_1.keys())[0]  # 只针对单输入
        # dtype = self._trt_to_np_dtype(self.engine.get_tensor_dtype(input_name))
        # x = x.astype(dtype)
        # np.copyto(self.inputs_1[input_name].host, x.ravel())

        # for inp in self.inputs_1.values():
        #     cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # for out in self.outputs_1.values():
        #     cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        # self.stream.synchronize()
        # outputs = {}
        # for key, out in self.outputs_1.items():
        #     outputs[key] = out.host.reshape((self.max_batch_size, *self.output_dims[key]))

        # return outputs

    def base_inference(self, infer_input):
        self.inputs[0]['host'] = infer_input
        # 2、将输入的数据同步到gpu上面，从 host -> device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 3、执行推理（Execute / Executev2）
        # execute_async_v2  ： 对批处理异步执行推理。此方法仅适用于从没有隐式批处理维度的网络构建的执行上下文。
        # execute_v2：      ： 在批次上同步执行推理。此方法仅适用于从没有隐式批处理维度的网络构建的执行上下文。
        # 同步和异步的差异    ： 在同一个上下文管理器中，程序的执行是否严格按照从上到下的过程。
        #                     如，连续输入多张图片，同步 会等处理完结果再去获得下一张，异步会开启多线程，提前处理数据
        self.cfx.push()
        self.context.execute_async_v2(
            bindings=self.bindings,  # 要进行推理的数据，放进去的时候，只有输入，出来输入、输出都有了
            stream_handle=self.stream.handle  # 将在其上执行推理内核的 CUDA 流的句柄。
        )
        self.cfx.pop()
        # 4、Buffer 拷贝操作	Device to Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # 5、将 stream 中的数据进行梳理
        self.stream.synchronize()

        # 6、整理输出
        infer_output = [out['host'].copy() for out in self.outputs]
        return infer_output
    
    def _load_engine(self) -> trt.ICudaEngine:
        # 2、加载 TRT 模型
        self.logger = trt.Logger(trt.Logger.INFO)
        if os.path.isfile(self.engine_file_path):
            self.engine = self._read_TRT_file(self.engine_file_path)
            assert self.engine, "从 TRT 文件中读取的 engine 为 None ! "
        elif (os.path.isfile(self.onnx_file_path)) and (self.engine_file_path is not None):
            self.engine = self._onnx_to_TRT_model(self.onnx_file_path, self.engine_file_path, self.precision_flop)
            assert self.engine, "从 onnx 文件中转换的 engine 为 None ! "
        else:
            raise Exception("请指定有效的模型权重路径")

    def _allocate_buffers(self):
        inputs = {}
        outputs = {}
        bindings = []
        self.stream = cuda.Stream()

        for idx, binding in enumerate(self.engine):
            dims = self.engine.get_tensor_shape(binding)[1:]
            size = trt.volume(dims) * self.dynamic_max_batch_size
            dtype = self._trt_to_np_dtype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(binding, (self.dynamic_max_batch_size, *dims))
                self.inputs.append({'host': host_mem, 'device': device_mem})
                self.inputs_1[binding] = {'host': host_mem, 'device': device_mem}
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                self.outputs_1[binding] = {'host': host_mem, 'device': device_mem}
                self.output_dims[binding] = dims

    def _trt_to_np_dtype(self, trt_dtype):  # 之前少了self
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
        
    def _read_TRT_file(self, engine_file_path):
        """从已经存在的文件中读取 TRT 模型

        Args:
            engine_file_path: 已经存在的 TRT 模型的路径

        Returns:
            加载完成的 engine
        """
        # 建立一个反序列化器
        runtime = trt.Runtime(self.logger)
        # 将路径转换为绝对路径防止出错
        engine_file_path = os.path.realpath(engine_file_path)
        # 判断TRT模型是否存在
        if not os.path.isfile(engine_file_path):
            print("模型文件：{}不存在".format(engine_file_path))
            return None
        # 反序列化TRT模型
        with open(engine_file_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine, "反序列化之后的引擎为空，确保转换过程的正确性."
        print("从{}成功载入引擎.".format(engine_file_path))

        return engine
    
    def _onnx_to_TRT_model(self, onnx_file_path, engine_file_path, precision_flop):
        """构建期 -> 转换网络模型为 TRT 模型

        Args:
            onnx_file_path  : 要转换的 onnx 模型的路径
            engine_file_path: 转换之后的 TRT engine 的路径
            precision_flop  : 转换过程中所使用的精度

        Returns:
            转化成功: engine
            转换失败: None
        """
        # ---------------------------------#
        # 准备全局信息
        # ---------------------------------#
        # 构建一个 构建器
        builder = trt.Builder(self.logger)
        builder.max_batch_size = 1

        # ---------------------------------#
        # 第一步，读取 onnx
        # ---------------------------------#
        # 1-1、设置网络读取的 flag
        # EXPLICIT_BATCH 相教于 IMPLICIT_BATCH 模式，会显示的将 batch 的维度包含在张量维度当中，
        # 有了 batch大小的，我们就可以进行一些必须包含 batch 大小的操作了，如 Layer Normalization。
        # 不然在推理阶段，应当指定推理的 batch 的大小。目前主流的使用的 EXPLICIT_BATCH 模式
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # 1-3、构建一个空的网络计算图
        network = builder.create_network(network_flags)
        # 1-4、将空的网络计算图和相应的 logger 设置装载进一个 解析器里面
        parser = trt.OnnxParser(network, self.logger)
        # 1-5、打开 onnx 压缩文件，进行模型的解析工作。
        # 解析器 工作完成之后，网络计算图的内容为我们所解析的网络的内容。
        onnx_file_path = os.path.realpath(onnx_file_path)  # 将路径转换为绝对路径防止出错
        if not os.path.isfile(onnx_file_path):
            print("onnx文件不存在，请检查onnx文件路径是否正确")
            return None
        else:
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("错误：解析onnx文件{}失败".format(onnx_file_path))
                    # 出错了，将相关错误的地方打印出来，进行可视化处理`-`
                    for error in range(parser.num_errors):
                        print(parser.num_errors, ": ", parser.get_error(error))
                    return None
            print("编译解析onnx文件成功")
        # 6、将转换之后的模型的输入输出的对应的大小进行打印，从而进行验证
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        print("描述网络输入输出情况")
        for inp in inputs:
            print("输入{}的维度为{}，数据类型为{}".format(inp.name, inp.shape, inp.dtype))
        for outp in outputs:
            print("输出{}的维度为{}，数据类型为{}".format(outp.name, outp.shape, outp.dtype))

        # ---------------------------------#
        # 第二步，转换为 TRT 模型
        # ---------------------------------#
        # 2-1、设置 构建器 的 相关配置器
        # 应当丢弃老版本的 builder. 进行设置的操作
        config = builder.create_builder_config()
        # 2-2、设置 可以为 TensorRT 提供策略的策略源。如CUBLAS、CUDNN 等
        # 也就是在矩阵计算和内存拷贝的过程中选择不同的策略
        # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
        # 2-3、给出模型中任一层能使用的内存上限，这里是 2^30,为 2GB
        # 每一层需要多少内存系统分配多少，并不是每次都分 2 GB
        config.max_workspace_size = 2 << 30
        # 2-4、设置 模型的转化精度
        if precision_flop == "FP32":
            # config.set_flag(trt.BuilderFlag.FP32)
            pass
        elif precision_flop == "FP16":
            if not builder.platform_has_fast_fp16:
                print("该平台/设备不支持 FP16")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
        elif precision_flop == "INT8":
            config.set_flag(trt.BuilderFlag.INT8)
        # 2-5，设置动态维度
        if len(self.dynamic_shapes) > 0:
            print("使用动态维度配置：{}".format(str(self.dynamic_shapes)))
            builder.max_batch_size = self.dynamic_max_batch_size
            profile = builder.create_optimization_profile()
            for binding_name, dynamic_shape in self.dynamic_shapes.items():
                min_shape, opt_shape, max_shape = dynamic_shape
                profile.set_shape(
                    binding_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
        # 2-6，从构建器构建引擎
        engine = builder.build_engine(network, config)
        print(2)

        # ---------------------------------#
        # 第三步，生成 SerializedNetwork
        # ---------------------------------#
        # 3-1、删除已经已经存在的版本
        engine_file_path = os.path.realpath(engine_file_path)  # 将路径转换为绝对路径防止出错
        if os.path.isfile(engine_file_path):
            try:
                os.remove(engine_file_path)
            except Exception:
                print("不能删除已存在的engine文件:{}".format(engine_file_path))
        print("创建engine文件:{}".format(engine_file_path))
        # 3-2、打开要写入的 TRT engine，利用引擎写入
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("onnx转trt成功，序列化engine文件保存在:{}".format(engine_file_path))

        return engine

