<!--
# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# PyTorch (LibTorch) Backend

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

The Triton backend for
[PyTorch](https://github.com/pytorch/pytorch)
is designed to run
[AOTInductor](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html)
compiled models using the PyTorch C++ API.
Models must be exported using `torch.export.export()` and compiled using `torch._inductor.aoti_compile_and_package()` to produce a `.pt2` package file.

You can learn more about Triton backends in the
[Triton Backend](https://github.com/triton-inference-server/backend)
repository.

Ask questions or report problems using
[Triton Server issues](https://github.com/triton-inference-server/server/issues).

Be sure to read all the information below as well as the
[general Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the [Triton Server](https://github.com/triton-inference-server/server) repository.

## Build the PyTorch Backend

Use a recent cmake to build.
First install the required dependencies.

```bash
apt-get install rapidjson-dev python3-dev python3-pip
pip3 install patchelf==0.17.2
```

An appropriate PyTorch container from [NVIDIA NGC Catalog](https://ngc.nvidia.com) must be used.
For example, to build a backend that uses the 23.04 version of the PyTorch container from NGC:

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:23.04-py3" ..
make install
```

The following required Triton repositories will be pulled and used in the build.
By default, the `main` head will be used for each repository but the listed CMake argument can be used to override the value.

* triton-inference-server/backend: `-DTRITON_BACKEND_REPO_TAG=[tag]`
* triton-inference-server/core: `-DTRITON_CORE_REPO_TAG=[tag]`
* triton-inference-server/common: `-DTRITON_COMMON_REPO_TAG=[tag]`

## Build the PyTorch Backend With Custom PyTorch

Currently, Triton requires that a specially patched version of PyTorch be used with the PyTorch backend.
The full source for these PyTorch versions are available as Docker images from
[NGC](https://ngc.nvidia.com).

For example, the PyTorch version compatible with the 25.09 release of Triton is available as `nvcr.io/nvidia/pytorch:25.09-py3` which supports PyTorch version `2.9.0a0`.

> [!NOTE]
> Additional details and version information can be found in the container's
> [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-09.html#rel-25-09).

Copy over the LibTorch and TorchVision headers and libraries from the
[PyTorch NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
into local directories.
You can see which headers and libraries are needed/copied from the docker.

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_INCLUDE_PATHS="<PATH_PREFIX>/torch;<PATH_PREFIX>/torch/torch/csrc/api/include;<PATH_PREFIX>/torchvision" -DTRITON_PYTORCH_LIB_PATHS="<LIB_PATH_PREFIX>" ..
make install
```

## Using the PyTorch Backend

### AOTInductor Models

This backend supports AOTInductor compiled models packaged as `.pt2` files. To create a model:

1. Export your model using `torch.export.export()`
2. Compile it using `torch._inductor.aoti_compile_and_package()`

Example Python code to generate the model:

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs = (torch.randn(8, 10, device=device),)

    # Optional: Specify dynamic dimensions
    batch_dim = torch.export.Dim("batch", min=1, max=1024)

    # Export the model
    exported = torch.export.export(
        model, example_inputs, dynamic_shapes={"x": {0: batch_dim}}
    )

    # Compile and package
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path="model.pt2",
    )
```

The model repository should look like:

```bash
model_repository/
`-- model_directory
    |-- 1
    |   `-- model.pt2
    `-- config.pbtxt
```

Where `model.pt2` is the AOTInductor compiled package.

> [!NOTE]
> AOTInductor models do not support string/bytes input or output types.
> All inputs and outputs must be tensor types.

## Configuration

Triton exposes some flags to control the execution mode of AOTInductor models through the `Parameters` section of the model's `config.pbtxt` file.

### Configuration Options

* `default_model_name`:
  Instructs the Triton PyTorch backend to load the model from a file of the given name.

  The model config specifying the option would look like:

  ```proto
  default_model_name: "another_file_name.pt2"
  ```

### Parameters

* `INFERENCE_MODE`:

  Boolean flag to enable the Inference Mode execution of PyTorch models.
  By default, the inference mode is enabled.

  [InferenceMode](https://pytorch.org/cppdocs/notes/inference_mode.html) is a RAII guard analogous to `NoGradMode` to be used when you are certain your operations will have no interactions with autograd.
  Compared to `NoGradMode`, code run under this mode gets better performance by disabling autograd.

  To enable inference mode, use the configuration example below:

  ```proto
  parameters: {
    key: "INFERENCE_MODE"
    value: { string_value: "true" }
  }
  ```

* `DISABLE_CUDNN`:

  Boolean flag to disable the cuDNN library.
  By default, cuDNN is enabled.

  [cuDNN](https://developer.nvidia.com/cudnn) is a GPU-accelerated library of primitives for deep neural networks.
  It provides highly tuned implementations for standard routines.

  Typically, models run with cuDNN enabled execute faster.
  However there are some exceptions where using cuDNN can be slower, cause higher memory usage, or result in errors.

  To disable cuDNN, use the configuration example below:

  ```proto
  parameters: {
    key: "DISABLE_CUDNN"
    value: { string_value: "true" }
  }
  ```

* `ENABLE_WEIGHT_SHARING`:

  Boolean flag to enable model instances on the same device to share weights.
  This optimization should not be used with stateful models.
  If not specified, weight sharing is disabled.

  To enable weight sharing, use the configuration example below:

  ```proto
  parameters: {
    key: "ENABLE_WEIGHT_SHARING"
    value: { string_value: "true" }
  }
  ```

* `ENABLE_CACHE_CLEANING`:

  Boolean flag to enable CUDA cache cleaning after each model execution.
  If not specified, cache cleaning is disabled.
  This flag has no effect if model is on CPU.

  Setting this flag to true will likely negatively impact the performance due to additional CUDA cache cleaning operation after each model execution.
  Therefore, you should only use this flag if you serve multiple models with Triton and encounter CUDA out-of-memory issues during model executions.

  To enable cleaning of the CUDA cache after every execution, use the configuration example below:

  ```proto
  parameters: {
    key: "ENABLE_CACHE_CLEANING"
    value: { string_value: "true" }
  }
  ```

* `INTER_OP_THREAD_COUNT`:

  PyTorch allows using multiple CPU threads during TorchScript model inference.
  One or more inference threads execute a model’s forward pass on the given inputs.
  Each inference thread invokes a JIT interpreter that executes the ops of a model inline, one by one.

  This parameter sets the size of this thread pool.
  The default value of this setting is the number of cpu cores.

  > [!TIP]
  > Refer to
  > [CPU Threading](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
  > on how to set this parameter properly.

  To set the inter-op thread count, use the configuration example below:

  ```proto
  parameters: {
    key: "INTER_OP_THREAD_COUNT"
    value: { string_value: "1" }
  }
  ```

> [!NOTE]
> This parameter is set globally for the PyTorch backend.
> The value from the first model config file that specifies this parameter will be used.
> Subsequent values from other model config files, if different, will be ignored.

* `INTRA_OP_THREAD_COUNT`:

  In addition to the inter-op parallelism, PyTorch can also utilize multiple threads within the ops (intra-op parallelism).
  This can be useful in many cases, including element-wise ops on large tensors, convolutions, GEMMs, embedding lookups and others.

  The default value for this setting is the number of CPU cores.

  > [!TIP]
  > Refer to
  > [CPU Threading](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
  > on how to set this parameter properly.

  To set the intra-op thread count, use the configuration example below:

  ```proto
  parameters: {
    key: "INTRA_OP_THREAD_COUNT"
    value: { string_value: "1" }
  }
  ```

### Model Instance Group Kind

The PyTorch backend supports the following kinds of
[Model Instance Groups](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
where the input tensors are placed as follows:

* `KIND_GPU`:

  Inputs are prepared on the GPU device associated with the model instance.

* `KIND_CPU`:

  Inputs are prepared on the CPU.

* `KIND_MODEL`:

  Inputs are prepared on the CPU.
  When loading the model, the backend does not choose the GPU device for the model;
  instead, it respects the device(s) specified in the model and uses them as they are during inference.

  This is useful when the model internally utilizes multiple GPUs, as demonstrated in
  [this example model](https://github.com/triton-inference-server/server/blob/main/qa/L0_libtorch_instance_group_kind_model/gen_models.py).

  > [!IMPORTANT]
  > If a device is not specified in the model, the backend uses the first available GPU device.

To set the model instance group, use the configuration example below:

```proto
instance_group {
   count: 2
   kind: KIND_GPU
}
```

### Customization

The following PyTorch settings may be customized by setting parameters on the
`config.pbtxt`.

[`torch.set_num_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html#torch.set_num_threads)

* Key: `NUM_THREADS`
* Value: The number of threads used for intra-op parallelism on CPU.

[`torch.set_num_interop_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html#torch.set_num_interop_threads)

* Key: `NUM_INTEROP_THREADS`
* Value: The number of threads used for interop parallelism on CPU.

For example:

```proto
parameters: {
  key: "NUM_THREADS"
  value: { string_value: "4" }
}
```

## Important Notes

* The execution of PyTorch model on GPU is asynchronous in nature.
  See
  [CUDA Asynchronous Execution](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
  for additional details.
  Consequently, an error in PyTorch model execution may be raised during the next few inference requests to the server.
  Setting environment variable `CUDA_LAUNCH_BLOCKING=1` when launching server will help in correctly debugging failing cases by forcing synchronous execution.

  * The PyTorch model in such cases may or may not recover from the failed state and a restart of the server may be required to continue serving successfully.

* AOTInductor models do not support string/bytes input or output types. All inputs and outputs must be tensor types.

* When using `KIND_MODEL` as model instance kind, the default device of the first parameter on the model is used.

* In a multi-GPU environment, ensure that the AOTInductor model was compiled for the correct device. By default, Triton creates a single execution instance of the model for each available GPU. You can explicitly specify the GPU device for the model instance in the
  [model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups).
