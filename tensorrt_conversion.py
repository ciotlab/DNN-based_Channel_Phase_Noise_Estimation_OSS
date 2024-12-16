import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import tensorrt as trt
from logzero import logger
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import onnx
import onnxruntime as ort


TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def save_simple_test_model(file_name):
    model = SimpleModel().eval()
    path = Path(__file__).parents[0].resolve() / 'torch_model'
    torch.save(model, path / (file_name + '.pt'))


def load_model(file_name):
    path = Path(__file__).parents[0].resolve() / 'torch_model'
    model = torch.load(path / (file_name + '.pt'))
    return model


def export_onnx_model(file_name, input_shape, batch_size):
    model = load_model(file_name).cuda().eval()
    inp = torch.randn(batch_size, *input_shape, dtype=torch.float32).cuda()
    path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.onnx')
    torch.onnx.export(
        model,
        inp,
        f=str(path),
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    logger.info("Export ONNX model successfully")


def build_and_save_trt_engine(file_name, input_shape, batch_size, fp16=False):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 50)  # Set the maximum workspace size for the builder
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.default_device_type = trt.DeviceType.GPU  # Set the default device type to GPU
        if fp16:
            config.flags |= 1 << int(trt.BuilderFlag.FP16)  # Set the optimization precision to float16
        else:
            config.flags &= 0 << int(trt.BuilderFlag.FP16)

        # Load ONNX model and parsing
        onnx_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.onnx')
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                logger.info('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))
                return None
        logger.info("ONNX parse ended")

        # Build TRT engine
        profile = builder.create_optimization_profile()
        network.add_input(name="input", dtype=trt.float32, shape=(-1, *input_shape))
        profile.set_shape(input="input", min=(1, *input_shape), opt=(batch_size, *input_shape),
                          max=(batch_size, *input_shape))
        config.add_optimization_profile(profile)
        logger.debug(f"config = {config}")
        logger.info("====================== building tensorrt engine... ====================")
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            logger.info('Tensorrt engine build failed.')
        else:
            logger.info('Tensorrt engine built successfully.')
            engine_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.engine')
            with open(engine_path, 'wb') as f:
                f.write(bytearray(engine))


def convert_pytorch_model_to_trt_engine(file_name, input_shape, batch_size, fp16=False):
    export_onnx_model(file_name=file_name, input_shape=input_shape, batch_size=batch_size)
    build_and_save_trt_engine(file_name=file_name, input_shape=input_shape, batch_size=batch_size, fp16=fp16)


class HostDeviceMem():
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


def trt_inference(file_name, input_tensor):
    with trt.Logger(trt.Logger.VERBOSE) as logger, trt.Runtime(logger) as runtime:
        engine_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.engine')
        with open(engine_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    inputs = []
    outputs = []
    bindings = []
    batch_size = input_tensor.shape[0]
    stream = cuda.Stream()
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)[1:]
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(tensor_name, input_tensor.shape)
        size = trt.volume((batch_size, *tensor_shape))
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
            output_shape = (batch_size, *tensor_shape)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer address to device bindings.
        # When cast to int, it's a linear index into the context's memory (like memory address).
        bindings.append(int(device_mem))

       # Append to the appropriate input/output list.
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    np.copyto(inputs[0].host, input_tensor.ravel())
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    context.execute_async_v3(stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    output_data = outputs[0].host.copy()
    output_data = np.reshape(output_data, output_shape)
    return output_data


def onnx_inference(file_name, input_tensor):
    onnx_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.onnx')
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    ort_sess = ort.InferenceSession(onnx_path)
    output = ort_sess.run(output_names=None, input_feed={'input': input_tensor})[0]
    return output


def torch_inference(file_name, input_tensor):
    input_tensor = torch.tensor(input_tensor).cuda()
    model = load_model(file_name).cuda()
    output = model(input_tensor)
    output = output.detach().cpu().numpy()
    return output


if __name__ == "__main__":
    # # Compile simple model
    # file_name = "simple_model"
    # input_shape = (10,)
    # batch_size = 16
    # save_simple_test_model(file_name)
    # convert_pytorch_model_to_trt_engine(file_name=file_name, input_shape=input_shape, batch_size=batch_size)

    # # Simple model inference test
    # file_name = "simple_model"
    # input_shape = (10,)
    # batch_size = 8
    # input_tensor = np.random.randn(batch_size, *input_shape).astype(np.float32)
    # output_trt = trt_inference(file_name, input_tensor)
    # output_onnx = onnx_inference(file_name, input_tensor)
    # output_torch = torch_inference(file_name, input_tensor)
    # pass

    # Compile transformer channel model
    file_name = "transformer_ch"
    input_shape = (2, 3072)
    batch_size = 1
    convert_pytorch_model_to_trt_engine(file_name=file_name, input_shape=input_shape, batch_size=batch_size,
                                        fp16=False)

    # Transformer channel model inference test
    file_name = "transformer_ch"
    input_shape = (2, 3072)
    batch_size = 1
    input_tensor = np.random.randn(batch_size, *input_shape).astype(np.float32)
    output_trt = trt_inference(file_name, input_tensor)
    output_onnx = onnx_inference(file_name, input_tensor)
    output_torch = torch_inference(file_name, input_tensor)
    pass

    # Compile transformer phase noise model
    file_name = "transformer_pn"
    input_shape = (2, 896)
    batch_size = 1
    convert_pytorch_model_to_trt_engine(file_name=file_name, input_shape=input_shape, batch_size=batch_size,
                                        fp16=False)

    # Transformer phase noise model inference test
    file_name = "transformer_pn"
    input_shape = (2, 896)
    batch_size = 1
    input_tensor = np.random.randn(batch_size, *input_shape).astype(np.float32)
    output_trt = trt_inference(file_name, input_tensor)
    output_onnx = onnx_inference(file_name, input_tensor)
    output_torch = torch_inference(file_name, input_tensor)
    pass