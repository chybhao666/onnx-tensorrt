# TensorRT backend for ONNX

Parses ONNX models for execution with [TensorRT](https://developer.nvidia.com/tensorrt).

See also the [TensorRT documentation](https://docs.nvidia.com/deeplearning/sdk/#inference).

## ONNX Python backend usage

The TensorRT backend for ONNX can be used in Python as follows:

```python
import onnx
import onnx_tensorrt.backend as backend
import pycuda.autoinit
import numpy as np

model = onnx.load("/path/to/model.onnx")
engine = backend.prepare(model, device='CUDA:1')
input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)
```

## Executable usage

ONNX models can be converted to serialized TensorRT engines using the `onnx2trt` executable:

    onnx2trt my_model.onnx -o my_engine.trt

ONNX models can also be converted to human-readable text:

    onnx2trt my_model.onnx -t my_model.onnx.txt

See more usage information by running:

    onnx2trt -h

## C++ library usage

The model parser library, libnvonnxparser.so, has a C++ API declared in this header:

    NvOnnxParser.h

TensorRT engines built using this parser must use the plugin factory provided in
libnvonnxparser_runtime.so, which has a C++ API declared in this header:

    NvOnnxParserRuntime.h

## Installation

### Dependencies

 - [Protobuf](https://github.com/google/protobuf/releases)
 - [TensorRT 4+](https://developer.nvidia.com/tensorrt)

### Download the code
Clone the code from GitHub. 

    git clone --recursive https://github.com/onnx/onnx-tensorrt.git

### Executable and libraries

Suppose your TensorRT library is located at `/opt/tensorrt`. Build the `onnx2trt` executable and the `libnvonnxparser*` libraries using CMake:

    mkdir build
    cd build
    cmake .. -DTENSORRT_ROOT=/opt/tensorrt
    make -j8
    sudo make install

### Python modules

Build the Python wrappers and modules by running:

    python setup.py build
    sudo python setup.py install

### Docker image (tested)

Build the onnx_tensorrt Docker by first copying TensorRT to the cloned `onnx-tensorrt` directory, and optionally downloading the PyTorch wheel for the respective Python version (the Dockerfile will install PyTorch if the wheel is present):

    cp /path/to/TensorRT-4.0.*.tar.gz .
    wget http://download.pytorch.org/whl/cu90/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl (Optional for Python 2 version, see below)
    wget http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl (Optional for Python 3, see below)

For the Python 2 version, run: 
    
    [sudo] docker build [--network=host] -t onnx_tensorrt_py2 -f Dockerfile.python2 .

For the Python 3 version, run:

    [sudo] docker build [--network=host] -t onnx_tensorrt_py3 -f Dockerfile.python3 .


Run a Docker container using above images for the Python 2 or Python 3 version:

    [sudo] nvidia-docker run -it [--net=host] -v $PWD:/shared onnx_tensorrt_[py2 or py3] /bin/bash

To run the [Jupyter notebook explaining how to use TensorRT's custom layer interface with ONNX](samples/onnx_custom_plugin.ipynb), run the command below (selecting the Python 2 or Python 3 version) and open the link shown in the console with your browser:
    
    [sudo] nvidia-docker run -it --net=host -v $PWD/samples:/workspace/samples onnx_tensorrt_[py2 or py3] jupyter notebook --allow-root /workspace/samples/onnx_custom_plugin.ipynb

### Tests

After installation (or inside the Docker container), ONNX backend tests can be run as follows:

Real model tests only:

    python onnx_backend_test.py OnnxBackendRealModelTest

All tests:

    python onnx_backend_test.py

## Pre-trained models

Pre-trained Caffe2 models in ONNX format can be found at https://github.com/onnx/models
