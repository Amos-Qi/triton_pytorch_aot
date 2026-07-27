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

### CUDA Graph Capture/Replay

For static-shape AOTInductor models, the backend can capture the whole forward
pass into CUDA graphs (one per request-batch size "bucket") and replay them,
eliminating per-launch CPU overhead. Requests are collected into fixed static
input buffers, the batch is padded up to the nearest captured bucket, the graph
replays, and each output's batch dimension is sliced back to the real request
count. Any batch that cannot replay (shape outside the buckets, capture
failure, degenerate inputs) transparently falls back to the eager AOTInductor
run.

* `ENABLE_CUDA_GRAPH`:

  Boolean flag to enable CUDA-graph capture/replay.
  If not specified, CUDA graphs are disabled and the eager path is unchanged.

  Enabling this builds the AOTInductor loader single-threaded with one runner
  (capture through a multi-runner loader is illegal), and forces
  `ENABLE_WEIGHT_SHARING` off: every instance needs its own single-runner
  loader and keeps its own captured graphs. On a CPU instance the flag is
  ignored with a warning (eager run, default threading).

  ```proto
  parameters: {
    key: "ENABLE_CUDA_GRAPH"
    value: { string_value: "true" }
  }
  ```

* `CUDA_GRAPH_BATCH_SIZES`:

  Comma-separated list of request-batch sizes (buckets) to capture graphs at,
  e.g. `"2,4,6"`. A batch of size `r` replays the smallest captured bucket
  `>= r` (padded with inert dummy requests whose output rows are discarded);
  `r` above the largest bucket runs eager.

  Semantics that matter:

  * **Omitting the parameter enables unbounded capture**: the backend captures
    one graph per first-seen REQUEST COUNT (the cache is keyed by `R` only),
    at whatever input widths that `R` first arrived with — if widths later
    change for the same `R`, no new graph is captured; the one-time warmup
    width self-heal is the only protection. Only use this when both the
    request-count universe and the input widths are known to be fixed.
  * **A present but malformed or empty value fails model load** (the
    parameter's presence declares the intent to bound capture; silently
    falling back to unbounded would invert the meaning). Every non-empty
    token must parse fully as an integer.
  * Entries outside `[1, max_batch_size]` are dropped with a warning; if that
    leaves the list empty, model load fails.
  * Bucket padding (and load-time warmup) only engage for models exposing the
    request-batch layout convention below; other models replay exact-size
    shapes only.

  ```proto
  parameters: {
    key: "CUDA_GRAPH_BATCH_SIZES"
    value: { string_value: "2,4,6" }
  }
  ```

* `CUDA_GRAPH_WARMUP_SINGLE_WIDTH` / `CUDA_GRAPH_WARMUP_MULTI_WIDTH`:

  Feature widths used to build synthetic zero-valued inputs so every
  configured bucket is captured at model load, BEFORE the instance goes READY
  (budget ~150 ms per bucket per instance; small models on partitioned GPUs
  have measured well under that). `SINGLE_WIDTH` is the per-request column
  count of the dense (single) input; `MULTI_WIDTH` is the per-request element
  count of the flattened ragged (multi) input.

  Both must be set (and a non-empty `CUDA_GRAPH_BATCH_SIZES` configured) or
  warmup is skipped and each bucket captures lazily on its first live request
  — captures run ~150 ms and flush the CUDA allocator cache, so lazy capture
  under production traffic is a latency cliff. If the configured widths do not
  match real traffic, the warmup captures are evicted on the first request and
  lazy capture self-heals at the true shape.

  ```proto
  parameters: {
    key: "CUDA_GRAPH_WARMUP_SINGLE_WIDTH"
    value: { string_value: "2756" }
  }
  parameters: {
    key: "CUDA_GRAPH_WARMUP_MULTI_WIDTH"
    value: { string_value: "157696" }
  }
  ```

#### Request-batch layout convention

Bucket padding rewrites the third input as a per-request cumulative element
count, which is only meaningful for models declaring exactly this layout:

| index | input name | shape |
|---|---|---|
| 0 | `packed_single_batch_tensor` | `(R, single_width)` |
| 1 | `packed_multiple_batch_tensor` | flattened ragged, `(R * multi_width,)` |
| 2 | `request_end_position` | `(R,)` accumulated element counts (batch input) |

Models that do not match this convention still get exact-size graph replay,
but never padding or load-time warmup. For models that DO match it, an empty
ragged request is rejected with `INVALID_ARG` before reaching the model: an
empty request otherwise trips a device-side assert inside the model that
poisons the CUDA context (the server keeps answering readiness while every
inference fails). The whole batch receives the error — a retryable failure,
unlike a bricked instance. Models outside the convention do not get this
guard (the backend cannot know an empty ragged input is invalid for them).

#### Metrics

When CUDA graphs are enabled the backend registers four Prometheus counters
(no-ops if the metrics API is unavailable):

| metric | labels | meaning |
|---|---|---|
| `cudagraph_replays_total` | `bucket` | successful graph replays |
| `cudagraph_pad_waste_rows_total` | `bucket` | dummy (padding) request rows replayed |
| `cudagraph_eager_fallbacks_total` | — | requests served eager instead of by a graph |
| `cudagraph_capture_failures_total` | `bucket` | failed capture attempts |

A healthy steady state is a high replay rate, near-zero eager fallbacks, and
zero capture failures after load.

#### Operational notes

* All of an instance's buckets share one capture memory pool, so graph memory
  scales with the largest bucket rather than the sum.
* Before each capture the backend calls
  `CUDACachingAllocator::emptyCache()` — a process-global cache flush with a
  device sync. This is what lets many instances' pools co-exist on one
  device, but in a multi-model server it briefly affects other models on the
  process; captures are also serialized model-wide for the same reason.
  Captures happen at load (warmup) or rarely (lazy/self-heal), so the impact
  window is small.
* A failed capture intentionally leaks the partial `at::cuda::CUDAGraph`
  object: its destructor can throw after a failed capture, which would
  terminate the process. The failure is negative-cached (the bucket pins to
  eager), so the leak is bounded to one object per failed ATTEMPT — at most
  two per bucket at load (the warmup's retry pass clears the negative cache
  once and re-attempts).

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
