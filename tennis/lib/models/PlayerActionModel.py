# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/03 18:22
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import onnxruntime as ort
from .AbstractModel import AbstractModel
from ..utils.process import keypoint_normalization_within_frame, update_action_counter
class PlayerActionModel(AbstractModel):

    def __init__(self, 
                 engine_file_path=None,
                 onnx_file_path=None,
                 use_onnx=True,
                 precision_flop="FP16",
                 dynamic_shapes={},
                 dynamic_max_batch_size=1,
                 window_size=30):
               
        """
        初始化动作分类模型

        :param engine_file_path: trt模型权重路径
        :param onnx_file_path: onnx模型权重路径
        :param use_onnx: 是否使用onnx模型
        :param precision_flop: 使用的精度类型，可选["FP32", "FP16", "int8"]
        :param dynamic_shapes: 自定义动态维度
        :param dynamic_max_batch_size: 动态batch size的最大值
        """
        '''
        kptseq_with_id: 关键点序列
        actionseq_with_id: 动作序列
        actioncounter_with_id: 动作计数
        action_prob_sequences_with_id: 动作概率序列
        action_timestamps_with_id: 动作时间戳
        window_size: 窗口大小
        '''
        self.kptseq_with_id = {}
        self.actionseq_with_id = {}
        self.actioncounter_with_id = {}
        self.action_prob_sequences_with_id = {}
        self.action_timestamps_with_id = {}
        self.window_size = window_size
        self.cfx = cuda.Device(0).make_context()
        self.engine_file_path = engine_file_path
        self.onnx_file_path = onnx_file_path
        self.use_onnx = use_onnx
        self.precision_flop = precision_flop
        self.dynamic_shapes = dynamic_shapes
        self.dynamic_max_batch_size = dynamic_max_batch_size
        self.inputs = []
        self.outputs = []
        self.bindings = []
        print("初始化动作分类模型")
        
    def init_model(self):
        print("初始化球场检测模型")

        """
        加载ONNX模型 
        """
        if self.use_onnx:
            # 初始化onnx模型
            self.ort_session = ort.InferenceSession(self.onnx_file_path, providers=['CUDAExecutionProvider'])
            return

        """
        加载 TRT 模型, 并加载一些多次推理过程共用的参数。
        情况 1、TRT 模型不存在，会先从输入的 onnx 模型创建一个 TRT 模型，并保存，再进行推导；
        情况 2、TRT 模型存在，直接进行推导
        """
        # 1、加载 logger 等级
        self.logger = trt.Logger(trt.Logger.ERROR)

        # 2、加载 TRT 模型
        if os.path.isfile(self.engine_file_path):
            self.engine = self._read_TRT_file(self.engine_file_path)
            assert self.engine, "从 TRT 文件中读取的 engine 为 None ! "
        elif (os.path.isfile(self.onnx_file_path)) and (self.engine_file_path is not None):
            self.engine = self._onnx_to_TRT_model(self.onnx_file_path, self.engine_file_path, self.precision_flop)
            assert self.engine, "从 onnx 文件中转换的 engine 为 None ! "
        else:
            raise Exception("请指定有效的模型权重路径")

        # 3、创建上下管理器，后面进行推导使用
        self.context = self.engine.create_execution_context()
        assert self.context, "创建的上下文管理器 context 为空，请检查相应的操作"

        # 4、创建数据传输流，在 cpu <--> gpu 之间传输数据的时候使用。
        self.stream = cuda.Stream()

        # 5、在 cpu 和 gpu 上申请内存
        for binding in self.engine:
            # 获取当前输入输出张量的数值个数
            dims = self.engine.get_tensor_shape(binding)[1:]
            if dims[-1] == -1:
                dims = (3, 320, 320)
            size = trt.volume(dims) * self.dynamic_max_batch_size
            # 获取当前输入输出张量的数值类型
            dtype = self._trt_to_np_dtype(self.engine.get_tensor_dtype(binding))
            # 数值个数 * 单个数值类型 = 内存的真实大小，先申请cpu上的内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配gpu上的内存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # 将设备缓冲区添加到设备绑定中
            self.bindings.append(int(device_mem))
            print("数值个数: {}, 数据类型: {}, 申请的gpu内存: {}".format(size, dtype, device_mem))
            # 区分输入的和输出申请的内存
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(binding, (self.dynamic_max_batch_size, *dims))
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

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
        
    def _pre_process(self):
        pass

    def inference(self, keypoints, primary_id, frame_id):
        # 初始化字典
        if primary_id not in self.kptseq_with_id:
            self.kptseq_with_id[primary_id] = []
        if primary_id not in self.actionseq_with_id:
            self.actionseq_with_id[primary_id] = [0 for _ in range(self.window_size-1)]
        if primary_id not in self.actioncounter_with_id:
            # 将其他id的动作计数累加到primary_id中
            self.actioncounter_with_id[primary_id] = {0: 0, 1: 0, 2: 0, 3: 0}
            for id, action_counter in self.actioncounter_with_id.items():
                if id != primary_id:
                    for action_type, count in action_counter.items():
                        self.actioncounter_with_id[primary_id][action_type] += count
        if primary_id not in self.action_prob_sequences_with_id:
            self.action_prob_sequences_with_id[primary_id] = {0: [0 for _ in range(self.window_size-1)], 1: [0 for _ in range(self.window_size-1)], 
                                                    2: [0 for _ in range(self.window_size-1)], 3: [0 for _ in range(self.window_size-1)]}
        if primary_id not in self.action_timestamps_with_id:
            self.action_timestamps_with_id[primary_id] = {0: [], 1: [], 2: [], 3: []}
            for id, action_timestamps in self.action_timestamps_with_id.items():
                if id != primary_id:
                    for action_type, timestamps in action_timestamps.items():
                        self.action_timestamps_with_id[primary_id][action_type].extend(timestamps)
        self.kptseq_with_id[primary_id].append(keypoints)
        
        # 处理窗口
        if len(self.kptseq_with_id[primary_id]) >= self.window_size:
            self.kptseq_with_id[primary_id] = self.kptseq_with_id[primary_id][-self.window_size:]
            current_window = np.array(self.kptseq_with_id[primary_id])
            infer_input = keypoint_normalization_within_frame(current_window).reshape(1, self.window_size, -1)
            
            if self.use_onnx:
                input_name = self.ort_session.get_inputs()[0].name
                output_name = self.ort_session.get_outputs()[0].name
                infer_output = self.ort_session.run([output_name], {input_name: infer_input})
            else:
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
            infer_output = infer_output[0]
            action_probs = infer_output.squeeze().tolist()
            for action_type, prob in enumerate(action_probs):
                self.action_prob_sequences_with_id[primary_id][action_type].append(prob)
                # action_prob_sequences_with_id_static[primary_id][action_type].append(prob)
            shot_type1 = np.argmax(infer_output)
            self.actionseq_with_id[primary_id].append(shot_type1)
        
        # 更新动作计数
        update_action_counter(self.action_prob_sequences_with_id, self.actioncounter_with_id, 20, self.action_timestamps_with_id, frame_id)
        self._post_process(primary_id)
        return self.actioncounter_with_id, self.action_timestamps_with_id
        
    def _post_process(self, primary_id):
        self.actioncounter_with_id = {primary_id:self.actioncounter_with_id[primary_id]}
        self.action_prob_sequences_with_id = {primary_id:self.action_prob_sequences_with_id[primary_id]}
        self.actionseq_with_id = {primary_id:self.actionseq_with_id[primary_id][-2:]}
        self.kptseq_with_id = {primary_id:self.kptseq_with_id[primary_id]}
        self.action_timestamps_with_id = {primary_id:self.action_timestamps_with_id[primary_id]}

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

    def __del__(self):
        # 清理资源
        self.context = None
        if self.engine is not None:
            self.engine = None
        if self.cfx:
            self.cfx.pop()  # 确保弹出上下文
            self.cfx.detach()  # 分离所有资源（如果在CUDA文档中推荐）
            self.cfx = None