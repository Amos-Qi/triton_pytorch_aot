// Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "model_instance_state.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "cuda_graph_bucketing.h"
#include "string_utils.hh"

#ifdef TRITON_PYTORCH_ENABLE_TORCHVISION
// Suppress warnings in torch headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning(push, 0)
#include <torchvision/ops/ops.h>
#include <torchvision/vision.h>  // Torchvision header
#pragma warning(pop)
#pragma GCC diagnostic pop
#endif  // TRITON_PYTORCH_ENABLE_TORCHVISION

#ifdef TRITON_ENABLE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU


namespace triton::backend::pytorch {

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), device_(torch::kCPU), device_cnt_(0)
{
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    device_ = torch::Device(torch::kCUDA, DeviceId());
    CreateCudaEvents(DeviceId());
    if (model_state->EnabledCudaGraph()) {
      // Persistent, timing-disabled event to order the graph-replay stream
      // behind the input-collection stream in ExecuteWithCudaGraph (the device
      // was already set by CreateCudaEvents above).
      THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
          cudaEventCreateWithFlags(
              &cuda_graph_input_ready_event_, cudaEventDisableTiming),
          TRITONSERVER_ERROR_INTERNAL,
          "Failed to create CUDA-graph input-ready event"));
      THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
          cudaEventCreateWithFlags(
              &cuda_graph_output_ready_event_, cudaEventDisableTiming),
          TRITONSERVER_ERROR_INTERNAL,
          "Failed to create CUDA-graph output-ready event"));
    }
#endif
  }

#ifdef TRITON_ENABLE_GPU
  device_cnt_ = torch::cuda::device_count();
#endif

  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), device_, &model_path_, Kind(), &aoti_model_));

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
#ifdef TRITON_ENABLE_GPU
    // Since we cannot determine the exact devices used by the model, we create
    // a CUDA stream for every available device to ensure proper synchronization
    // of CUDA streams. This approach may have implications when a timestamp is
    // captured on a device that is not used by the model. Currently, this issue
    // is addressed by synchronizing the CUDA streams before recording
    // timestamps to prevent timestamp skewing. However, in the future, any
    // modifications to the CUDA stream synchronization logic should be handled
    // with caution.
    for (int i = 0; i < device_cnt_; i++) {
      cudaStream_t stream;
      THROW_IF_BACKEND_INSTANCE_ERROR(
          CreateCudaStream(i, 0 /* cuda_stream_priority */, &stream));
      stream_vec_.push_back(stream);
    }
    if (!stream_vec_.empty()) {
      // Create CUDA events on the first device that will be used for collecting
      // inputs/outputs.
      CreateCudaEvents(0);
    }
#endif
  }

  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (model_state->ModelConfig().Find("batch_input", &config_batch_inputs)) {
      batch_input_count_ = config_batch_inputs.ArraySize();
      expected_input_cnt += batch_input_count_;
    }
  }

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  triton::common::TritonJson::Value sequence_batching;
  if (model_state->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    bool have_start, have_end, have_ready, have_corrid;
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_START", false /* required */,
        &have_start));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_END", false /* required */,
        &have_end));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_READY", false /* required */,
        &have_ready));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateTypedSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_CORRID", false /* required */,
        &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }
    // Add the state inputs to the expected count
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      expected_input_cnt += states.ArraySize();
    }
  }
  supports_batching_ = model_state_->MaxBatchSize() > 0;

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

ModelInstanceState::~ModelInstanceState()
{
  aoti_model_.reset();
  ClearCache();

#ifdef TRITON_ENABLE_GPU
  if (cuda_graph_input_ready_event_ != nullptr) {
    LOG_IF_ERROR(
        ConvertCUDAStatusToTritonError(
            cudaEventDestroy(cuda_graph_input_ready_event_),
            TRITONSERVER_ERROR_INTERNAL,
            "Failed to destroy CUDA-graph input-ready event"),
        "~ModelInstanceState error: ");
    cuda_graph_input_ready_event_ = nullptr;
  }
  if (cuda_graph_output_ready_event_ != nullptr) {
    LOG_IF_ERROR(
        ConvertCUDAStatusToTritonError(
            cudaEventDestroy(cuda_graph_output_ready_event_),
            TRITONSERVER_ERROR_INTERNAL,
            "Failed to destroy CUDA-graph output-ready event"),
        "~ModelInstanceState error: ");
    cuda_graph_output_ready_event_ = nullptr;
  }
#endif

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
#ifdef TRITON_ENABLE_GPU
    for (size_t i = 0; i < stream_vec_.size(); i++) {
      LOG_IF_ERROR(
          ConvertCUDAStatusToTritonError(
              cudaSetDevice(i), TRITONSERVER_ERROR_INTERNAL,
              "Failed to set the device"),
          "Failed to set the device");

      LOG_IF_ERROR(
          ConvertCUDAStatusToTritonError(
              cudaStreamDestroy(stream_vec_[i]), TRITONSERVER_ERROR_INTERNAL,
              "Failed to destroy cuda stream"),
          "~ModelInstanceState error: ");
      stream_vec_[i] = nullptr;
    }
#endif
  }
}

void
ModelInstanceState::AddInputToMap(
    NamingConvention naming_convention, const std::string& io_name,
    const uint32_t index)
{
  std::string deliminator = "__";

  switch (naming_convention) {
    case NamingConvention::NAMED_INDEX: {
      int start_pos = io_name.find(deliminator);
      int ip_index = std::atoi(io_name.substr(start_pos + 2).c_str());
      input_index_map_[io_name] = ip_index;
      return;
    }
    case NamingConvention::FORWARD_ARGUMENT:
    case NamingConvention::STRICT_CONFIG_ORDERING: {
      input_index_map_[io_name] = index;
      return;
    }
  }
}

void
ModelInstanceState::ClearCache()
{
#ifdef TRITON_ENABLE_GPU
  if (device_.is_cuda() ||
      ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0))) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#endif  // TRITON_ENABLE_GPU
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

#ifdef TRITON_ENABLE_GPU
  // Capture the configured CUDA-graph buckets before the instance goes READY
  // (no-op unless ENABLE_CUDA_GRAPH + warmup widths + a bucket set are all
  // configured). WarmupCudaGraphs contains per-bucket failures itself; the
  // catch here is a belt for anything else (this function is called from an
  // extern "C" entry point, where an escaping exception kills the process).
  // Lazy capture is the safety net either way.
  try {
    (*state)->WarmupCudaGraphs();
  }
  catch (const std::exception& ex) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("CUDA-graph warmup threw outside the per-bucket "
                     "containment (") +
         ex.what() + "); continuing without warmup captures")
            .c_str());
  }
#endif

  return nullptr;  // success
}

void
ModelInstanceState::CreateCudaEvents(const int32_t& device_id)
{
#ifdef TRITON_ENABLE_GPU
  // Need to set the CUDA context so that the context that events are
  // created on match with contexts that events are recorded with.
  THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
      cudaSetDevice(device_id), TRITONSERVER_ERROR_INTERNAL,
      "Failed to set the device"));
  THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
      cudaEventCreate(&compute_input_start_event_), TRITONSERVER_ERROR_INTERNAL,
      "Failed to create cuda event"));
  THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
      cudaEventCreate(&compute_infer_start_event_), TRITONSERVER_ERROR_INTERNAL,
      "Failed to create cuda event"));
  THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
      cudaEventCreate(&compute_output_start_event_),
      TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
#endif
}

void
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count, std::vector<torch::Tensor>* input_tensors,
    std::vector<torch::Tensor>* output_tensors)
{
  NVTX_RANGE(nvtx_, "Execute " + Name());

  try {
    // enable/disable inference mode - supersedes NoGradGuard
    torch::InferenceMode infer_guard(model_state_->EnabledInferenceMode());

    // enable/disable cudnn
    at::globalContext().setUserEnabledCuDNN(model_state_->EnabledCudnn());

    torch::NoGradGuard no_grad;

#ifdef TRITON_ENABLE_GPU
    // Whole-forward CUDA-graph path (opt-in via ENABLE_CUDA_GRAPH, GPU only):
    // capture-on-first-use + replay for a fixed input shape. Falls through to
    // the eager run below if capture is not possible for these inputs.
    if (model_state_->EnabledCudaGraph() && !device_.is_cpu()) {
      if (ExecuteWithCudaGraph(input_tensors, output_tensors)) {
        return;
      }
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          "CUDA-graph execution unavailable for this input shape; falling back "
          "to eager AOTInductor run.");
    }
#endif

    // Run the AOTInductor model (launches kernels asynchronously)
    std::vector<torch::Tensor> model_outputs = aoti_model_->run(*input_tensors);

#ifdef TRITON_ENABLE_GPU
    // Synchronize the stream to wait for kernels to complete
    // This is necessary because aoti_model_->run() launches kernels
    // asynchronously
    if (!device_.is_cpu()) {
      cudaStreamSynchronize(GetCudaStreamByInstanceKind());
    }
#endif

    // Copy outputs to the output vector
    for (auto& output : model_outputs) {
      output_tensors->push_back(output);
    }
  }
  catch (std::exception& ex) {
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("PyTorch AOTInductor execute failure: " + std::string(ex.what()))
                .c_str()));
  }
}

float
ModelInstanceState::GetCudaEventElapsedTime(
    const cudaEvent_t& start_event, const cudaEvent_t& end_event)
{
  float duration = 0;
#ifdef TRITON_ENABLE_GPU
  // [FIXME] in the case of cudaEventElapsedTime failure, should handle
  // stats reporting more gracefully as the durations are inaccurate
  LOG_IF_ERROR(
      ConvertCUDAStatusToTritonError(
          cudaEventElapsedTime(&duration, start_event, end_event),
          TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time"),
      "Failed to capture elapsed time");
#endif
  return duration;
}


cudaStream_t
ModelInstanceState::GetCudaStreamByInstanceKind()
{
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    return stream_;
  } else if (
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) &&
      !stream_vec_.empty()) {
    return stream_vec_[0];
  }
#endif
  return nullptr;
}

#ifdef TRITON_ENABLE_GPU
bool
ModelInstanceState::HasRequestBatchLayout()
{
  if (request_batch_layout_state_ == 0) {
    const auto at_index = [&](const char* name, int idx) {
      const auto it = input_index_map_.find(name);
      return (it != input_index_map_.end()) && (it->second == idx);
    };
    request_batch_layout_state_ =
        (at_index("packed_single_batch_tensor", 0) &&
         at_index("packed_multiple_batch_tensor", 1) &&
         at_index("request_end_position", 2))
            ? 1
            : -1;
    if (request_batch_layout_state_ != 1 && model_state_->EnabledCudaGraph()) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Model instance '") + Name() +
           "' does not expose the canonical request-batch layout "
           "(packed_single/packed_multiple/request_end_position at inputs "
           "0/1/2); CUDA graphs will replay exact-R shapes only (no bucket "
           "padding, no load-time warmup)")
              .c_str());
    }
  }
  return request_batch_layout_state_ == 1;
}

std::string
ModelInstanceState::InputShapeKey(
    const std::vector<torch::Tensor>& inputs) const
{
  // Shape-only key: the AOTI .pt2 is static-shape, so a given (R, bucket) maps
  // to exactly one captured graph. dtypes/order are fixed for a model.
  std::string key;
  for (const auto& t : inputs) {
    for (const auto d : t.sizes()) {
      key += std::to_string(d);
      key += ',';
    }
    key += '|';
  }
  return key;
}

std::vector<torch::Tensor>
ModelInstanceState::PadRequestsUp(
    const std::vector<torch::Tensor>& inputs, int64_t r, int64_t bucket) const
{
  // request-batch layout: [0] packed_single (R, F1), [1] packed_multiple
  // flat (R*per,), [2] request_end_position (R,) = cumsum of per-request
  // element counts. Pad R -> bucket by appending zero rows to the packed
  // tensors (dummy requests are inert -- candidate queries attend only to their
  // own request's user-history KV, never across requests; their output rows are
  // sliced off), and regenerate the cumsum (a zero-padded cumsum would be
  // wrong).
  std::vector<torch::Tensor> out;
  out.reserve(inputs.size());
  // Callers guarantee these (r>0 eager-fallback guard + the canonical-layout
  // gate in ExecuteWithCudaGraph); a violation here would corrupt the padded
  // batch, so fail the request instead.
  TORCH_CHECK(
      r > 0 && bucket >= r && inputs.size() == 3 &&
          inputs[1].numel() % r == 0 && inputs[0].size(0) % r == 0,
      "PadRequestsUp preconditions violated (r=", r, ", bucket=", bucket,
      ", inputs=", inputs.size(), ")");
  const int64_t per =
      inputs[1].numel() /
      r;  // elements/request (uniform: candidates padded to bucket)
  for (size_t i = 0; i < inputs.size(); ++i) {
    const torch::Tensor& t = inputs[i];
    if (i == 2) {
      auto rep =
          torch::arange(
              1, bucket + 1,
              torch::TensorOptions().dtype(torch::kLong).device(t.device())) *
          per;
      out.push_back(rep.to(t.scalar_type()));
    } else {
      auto sizes = t.sizes().vec();
      sizes[0] =
          sizes[0] / r *
          bucket;  // scale leading dim R->bucket (packed_single; flat multiple)
      auto padded = torch::zeros(sizes, t.options());
      padded.narrow(0, 0, t.size(0)).copy_(t);
      out.push_back(padded);
    }
  }
  return out;
}

bool
ModelInstanceState::CaptureBucket(
    const std::vector<torch::Tensor>& inputs_at_bucket, int64_t bucket)
{
  // One graph per R bucket (all r that pad to the same bucket share it).
  const std::string key = "R=" + std::to_string(bucket);
  // We capture on a dedicated pool stream; restore the caller's stream on exit.
  const c10::cuda::CUDAStream prev_stream =
      c10::cuda::getCurrentCUDAStream(device_.index());

  // Raw pointer so a FAILED capture can LEAK the graph: after a failed capture
  // at::cuda::CUDAGraph's destructor can throw a second error ->
  // std::terminate, so we must not let it run. A capture failure is a rare
  // safety-net path (the intended static shape captures cleanly).
  at::cuda::CUDAGraph* graph = nullptr;
  // Hoisted out of the try so the failure path can end a dangling capture on
  // it.
  std::optional<c10::cuda::CUDAStream> capture_stream;
  // One capture at a time model-wide: concurrent captures gain nothing (each
  // instance keeps its own graphs) and their transient warmup allocations +
  // cache flushes race the VRAM margin captures need on a packed device.
  std::lock_guard<std::mutex> capture_lk(model_state_->CudaGraphCaptureMutex());
  try {
    // Low-level CUDA runner: run_with_cuda_stream runs AOTI on the capture
    // stream (the pattern proven by the route-c de-risk spike). Requires the
    // loader to have been built run_single_threaded (ENABLE_CUDA_GRAPH).
    auto* runner = static_cast<torch::inductor::AOTIModelContainerRunnerCuda*>(
        aoti_model_->get_runner());
    if (runner == nullptr) {
      throw std::runtime_error(
          "AOTI CUDA runner unavailable (loader not run_single_threaded?)");
    }

    // Dedicated stream so AOTI's caching-allocator scratch is captured on it.
    c10::cuda::CUDAStream stream =
        c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device_.index());
    capture_stream = stream;
    c10::cuda::setCurrentCUDAStream(stream);

    // Fixed-address input buffers (the graph replays into the same addresses),
    // at the bucket shape.
    std::vector<torch::Tensor> static_inputs;
    static_inputs.reserve(inputs_at_bucket.size());
    for (const auto& t : inputs_at_bucket) {
      static_inputs.push_back(t.clone());
    }

    // Warm up (allocate AOTI workspaces / autotune) before capture.
    for (int i = 0; i < 3; ++i) {
      (void)runner->run_with_cuda_stream(static_inputs, stream);
    }
    stream.synchronize();

    // Return the warmup runs' cached activation blocks to CUDA before the
    // capture allocates its pool: with several shared-device instances the
    // graph pools fit only if the allocator's cache isn't sitting on the
    // margin. cudaFree device-syncs, but capture is already a slow path
    // (load-time warmup, or a rare lazy capture).
    c10::cuda::CUDACachingAllocator::emptyCache();

    // Capture. Relaxed mode matches the proven spike. All of this instance's
    // buckets share ONE memory pool (only one graph replays at a time per
    // instance), so graph memory scales with the largest bucket, not the sum.
    graph = new at::cuda::CUDAGraph();
    graph->capture_begin(cuda_graph_mempool_, cudaStreamCaptureModeRelaxed);
    std::vector<torch::Tensor> static_outputs =
        runner->run_with_cuda_stream(static_inputs, stream);
    graph->capture_end();
    stream.synchronize();

    if (cuda_graph_mempool_ == at::cuda::MempoolId_t{0, 0}) {
      cuda_graph_mempool_ = graph->pool();
    }

    CudaGraphEntry entry{
        std::unique_ptr<at::cuda::CUDAGraph>(graph), std::move(static_inputs),
        std::move(static_outputs), stream};
    cuda_graph_cache_.emplace(key, std::move(entry));

    c10::cuda::setCurrentCUDAStream(prev_stream);
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Captured CUDA graph for bucket ") + key + " (shape " +
         InputShapeKey(inputs_at_bucket) + ") on model instance '" + Name() +
         "'")
            .c_str());
    return true;
  }
  catch (const std::exception& ex) {
    // Best-effort: pull the caching allocator out of capture mode, then LEAK
    // the graph (don't delete -> its dtor never runs). Fall back to eager.
    if (graph != nullptr) {
      try {
        graph->capture_end();
      }
      catch (...) {
      }
    }
    // capture_end() itself can throw mid-cleanup (e.g. the OOM that failed the
    // capture) and leave the POOL STREAM still in capture mode. A later capture
    // drawing that stream from the pool would have its setup ops captured
    // instead of executed -- garbage inputs, device-side asserts, poisoned
    // context, std::terminate from a throwing destructor, pod crashloop
    // (observed as a live crashloop). Verify at the raw CUDA level and
    // force-end any dangling capture, then clear the sticky error.
    if (capture_stream.has_value()) {
      cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(capture_stream->stream(), &cap_status) ==
              cudaSuccess &&
          cap_status != cudaStreamCaptureStatusNone) {
        cudaGraph_t dangling = nullptr;
        cudaStreamEndCapture(capture_stream->stream(), &dangling);
        if (dangling != nullptr) {
          cudaGraphDestroy(dangling);
        }
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            (std::string("Force-ended a dangling stream capture after the "
                         "failed capture of shape '") +
             key + "'")
                .c_str());
      }
    }
    cudaGetLastError();  // reset the sticky error from the failed capture
    cuda_graph_failed_.insert(key);  // don't retry this shape (negative cache)
    c10::cuda::setCurrentCUDAStream(prev_stream);
    model_state_->CudaGraphMetricCaptureFailure(bucket);
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("CUDA-graph capture failed (") + ex.what() +
         "); falling back to eager (and pinning eager) for shape '" + key + "'")
            .c_str());
    return false;
  }
}

bool
ModelInstanceState::ExecuteWithCudaGraph(
    std::vector<torch::Tensor>* input_tensors,
    std::vector<torch::Tensor>* output_tensors)
{
  // Only the 3-input request-batch layout is R-bucketed; any other
  // layout -> eager.
  if (input_tensors->size() != 3) {
    model_state_->CudaGraphMetricEagerFallback();
    return false;
  }
  // Request count R = smallest leading dim (packed_single &
  // request_end_position are (R,...); packed_multiple is (R*per,), larger).
  int64_t r = (*input_tensors)[0].size(0);
  for (const auto& t : *input_tensors) {
    r = std::min<int64_t>(r, t.size(0));
  }
  // Degenerate batch (e.g. an empty ragged input -> a 0-sized leading dim):
  // nothing can replay at R=0, and the per-request width math below divides
  // by r (a plain integer division -- r=0 would be fatal, not an exception).
  // Run it eager.
  if (r <= 0) {
    model_state_->CudaGraphMetricEagerFallback();
    return false;
  }

  // One-time width self-heal: warmup captured graphs at the configured widths
  // (CUDA_GRAPH_WARMUP_*_WIDTH). The cache is keyed only by "R=<bucket>", so if
  // the real traffic width differs, a wrong-shape warmup entry would be
  // replayed and copy_ would shape-mismatch every request -- evict all warmup
  // captures here so lazy capture re-populates at the real shape.
  if (!warmup_widths_checked_) {
    warmup_widths_checked_ = true;
    const int64_t cfg_single = model_state_->CudaGraphWarmupSingleWidth();
    const int64_t cfg_multi = model_state_->CudaGraphWarmupMultiWidth();
    if (cfg_single > 0 && cfg_multi > 0 && r > 0 &&
        (*input_tensors)[0].dim() >= 2) {
      const int64_t real_single = (*input_tensors)[0].size(1);
      const int64_t real_multi = (*input_tensors)[1].numel() / r;
      if (real_single != cfg_single || real_multi != cfg_multi) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            (std::string(
                 "CUDA-graph warmup WIDTH MISMATCH on model instance '") +
             Name() +
             "': configured single_width=" + std::to_string(cfg_single) +
             " multi_width=" + std::to_string(cfg_multi) +
             " but actual traffic single_width=" + std::to_string(real_single) +
             " multi_width=" + std::to_string(real_multi) +
             "; evicting warmup captures and re-capturing lazily at the real "
             "shape")
                .c_str());
        cuda_graph_cache_.clear();
        cuda_graph_failed_.clear();
        // The shared capture pool died with the graphs above; a fresh pool is
        // created on the next capture.
        cuda_graph_mempool_ = {0, 0};
        // Defensive: a prestaged pointer would dangle after clear(). It cannot
        // actually be set here (prestaging requires the batch widths to MATCH
        // the captured buffers, and this branch fires only on width mismatch),
        // but null it anyway; the narrowed input views keep their storage
        // alive via refcounting, so the slow path below still works.
        prestaged_entry_ = nullptr;
        prestaged_bucket_ = -1;
      }
    }
  }

  // Fast path: SetInputTensors already collected this batch DIRECTLY into an
  // existing bucket entry's static input buffers (and zeroed the padded tail),
  // so skip selection + PadRequestsUp + the static copy_ pass and replay.
  if (prestaged_entry_ != nullptr) {
    CudaGraphEntry& e = *prestaged_entry_;
    const int64_t bucket = prestaged_bucket_;
    prestaged_entry_ = nullptr;
    prestaged_bucket_ = -1;
    return ReplayCudaGraphEntry(
        e, r, bucket, /*copy_from=*/nullptr, output_tensors);
  }

  const auto& buckets = model_state_->CudaGraphBatchSizes();

  // Select an R bucket we can replay at. For the allowlist, advance ascending
  // from lower_bound(r), skipping buckets whose capture failed;
  // capture-on-first-use and, if that capture fails, advance to the next
  // healthy bucket. Bounded by the bucket-set size (no retry storm). Empty
  // allowlist keeps the "capture at the first-seen request count" behavior
  // (keyed by R only; widths are whatever that R first arrived with).
  std::vector<torch::Tensor> padded_storage;
  const std::vector<torch::Tensor>* inputs_at_bucket = nullptr;
  int64_t bucket = -1;
  std::string key;
  std::unordered_map<std::string, CudaGraphEntry>::iterator it =
      cuda_graph_cache_.end();

  if (buckets.empty()) {
    bucket = r;
    key = "R=" + std::to_string(bucket);
    // Negative cache: a bucket that failed capture once goes straight to eager.
    if (cuda_graph_failed_.find(key) != cuda_graph_failed_.end()) {
      model_state_->CudaGraphMetricEagerFallback();
      return false;
    }
    inputs_at_bucket = input_tensors;
    it = cuda_graph_cache_.find(key);
    if (it == cuda_graph_cache_.end()) {
      if (!CaptureBucket(*inputs_at_bucket, bucket)) {
        model_state_->CudaGraphMetricEagerFallback();
        return false;
      }
      it = cuda_graph_cache_.find(key);
    }
  } else {
    bool ready = false;
    for (auto b_it = buckets.lower_bound(r); b_it != buckets.end(); ++b_it) {
      const int64_t cand = *b_it;
      const std::string cand_key = "R=" + std::to_string(cand);
      // Skip buckets already known-bad (negative cache).
      if (cuda_graph_failed_.find(cand_key) != cuda_graph_failed_.end()) {
        continue;
      }
      // Pad R up to this candidate bucket (no-op if the batch already equals
      // it).
      std::vector<torch::Tensor> cand_padded;
      const std::vector<torch::Tensor>* cand_inputs = input_tensors;
      if (cand != r) {
        if (!HasRequestBatchLayout()) {
          // Padding rewrites input[2] as a per-request cumsum -- only defined
          // for the canonical layout. The set is ascending, so every later
          // bucket also needs padding: stop here and run eager.
          break;
        }
        cand_padded = PadRequestsUp(*input_tensors, r, cand);
        cand_inputs = &cand_padded;
      }
      auto cand_it = cuda_graph_cache_.find(cand_key);
      if (cand_it == cuda_graph_cache_.end()) {
        // Capture on first use; on failure advance to the next healthy bucket.
        if (!CaptureBucket(*cand_inputs, cand)) {
          continue;
        }
        cand_it = cuda_graph_cache_.find(cand_key);
      }
      bucket = cand;
      key = cand_key;
      padded_storage = std::move(cand_padded);
      inputs_at_bucket = (cand == r) ? input_tensors : &padded_storage;
      it = cand_it;
      ready = true;
      break;
    }
    if (!ready) {
      // R above the largest bucket, or every candidate bucket failed capture ->
      // eager.
      model_state_->CudaGraphMetricEagerFallback();
      return false;
    }
  }

  // ---- Replay (also runs the just-captured graph on first use) ----
  return ReplayCudaGraphEntry(
      it->second, r, bucket, inputs_at_bucket, output_tensors);
}

bool
ModelInstanceState::ReplayCudaGraphEntry(
    CudaGraphEntry& e, int64_t r, int64_t bucket,
    const std::vector<torch::Tensor>* copy_from,
    std::vector<torch::Tensor>* output_tensors)
{
  const c10::cuda::CUDAStream prev_stream =
      c10::cuda::getCurrentCUDAStream(device_.index());
  try {
    c10::cuda::setCurrentCUDAStream(e.stream);
    // The input tensors were produced on the instance stream (SetInputTensors);
    // make the graph stream wait on that producer before we touch the static
    // buffers (risk #9) via a lightweight event instead of a full-device sync
    // so other streams aren't stalled. The post-replay e.stream.synchronize()
    // below still gates outputs. The event also orders PRESTAGED writes (direct
    // collector H2D + tail zeroing, both on the instance stream).
    cudaEventRecord(
        cuda_graph_input_ready_event_, GetCudaStreamByInstanceKind());
    cudaStreamWaitEvent(e.stream.stream(), cuda_graph_input_ready_event_, 0);
    if (copy_from != nullptr) {
      for (size_t i = 0; i < copy_from->size(); ++i) {
        e.static_inputs[i].copy_((*copy_from)[i]);
      }
    }
    e.graph->replay();
    // Order the instance stream behind the replay via an event instead of a
    // host-blocking sync: the responder's output copies (and the next batch's
    // input collection) are issued on the instance stream, so they wait for
    // the replay on-device while the host proceeds straight to response
    // preparation. ReadOutputTensors touches only tensor metadata until its
    // Finalize gate (cudaStreamQuery + sync) confirms completion before the
    // responses send.
    cudaEventRecord(cuda_graph_output_ready_event_, e.stream.stream());
    cudaStreamWaitEvent(
        GetCudaStreamByInstanceKind(), cuda_graph_output_ready_event_, 0);

    // Slice each output's batch dim (dim 0 = R) back to the real request
    // count -- the padded (dummy) request rows are discarded. Views, not
    // clones: batches are serialized per instance, and the responder finishes
    // copying out of the static buffers (its Finalize sync) before this
    // request cycle ends, so the next replay cannot race them.
    for (const auto& o : e.static_outputs) {
      output_tensors->push_back(r == bucket ? o : o.narrow(0, 0, r));
    }

    c10::cuda::setCurrentCUDAStream(prev_stream);
    model_state_->CudaGraphMetricReplay(bucket);
    if (bucket > r) {
      model_state_->CudaGraphMetricPadWaste(bucket, PadWasteRows(r, bucket));
    }
    return true;
  }
  catch (const std::exception& ex) {
    c10::cuda::setCurrentCUDAStream(prev_stream);
    model_state_->CudaGraphMetricEagerFallback();
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("CUDA-graph replay failed (") + ex.what() +
         "); falling back to eager for bucket R=" + std::to_string(bucket))
            .c_str());
    return false;
  }
}

ModelInstanceState::CudaGraphEntry*
ModelInstanceState::FindReadyCudaGraphEntry(int64_t r, int64_t* bucket_out)
{
  if (!model_state_->EnabledCudaGraph() || device_.is_cpu()) {
    return nullptr;
  }
  const auto& buckets = model_state_->CudaGraphBatchSizes();
  if (buckets.empty()) {
    // Exact-shape mode: only a previously captured R=r entry qualifies.
    auto it = cuda_graph_cache_.find("R=" + std::to_string(r));
    if (it == cuda_graph_cache_.end()) {
      return nullptr;
    }
    *bucket_out = r;
    return &it->second;
  }
  for (auto b_it = buckets.lower_bound(r); b_it != buckets.end(); ++b_it) {
    const std::string key = "R=" + std::to_string(*b_it);
    if (cuda_graph_failed_.find(key) != cuda_graph_failed_.end()) {
      continue;
    }
    if (*b_it != r && !HasRequestBatchLayout()) {
      // The slow path never pads without the canonical layout (it goes eager
      // at the first bucket that would need padding); mirror that here.
      return nullptr;
    }
    auto it = cuda_graph_cache_.find(key);
    if (it == cuda_graph_cache_.end()) {
      // The slow path would capture-on-first-use at THIS bucket; defer to it
      // so prestaging never diverges from the canonical bucket selection.
      return nullptr;
    }
    *bucket_out = *b_it;
    return &it->second;
  }
  return nullptr;
}

void
ModelInstanceState::WarmupCudaGraphs()
{
  // Load-time warmup: capture every configured R bucket now (before the
  // instance goes READY) so the first live request of each bucket replays
  // instead of paying the ~150 ms capture cost inline. Zero-valued inputs are
  // safe -- capture depends on shapes, not values. Lazy capture in
  // ExecuteWithCudaGraph is the safety net for any bucket skipped or failed
  // here.
  if (!(model_state_->EnabledCudaGraph() && !device_.is_cpu() &&
        model_state_->CudaGraphWarmupSingleWidth() > 0 &&
        model_state_->CudaGraphWarmupMultiWidth() > 0 &&
        !model_state_->CudaGraphBatchSizes().empty())) {
    return;
  }
  // The synthetic zero inputs below are the canonical request-batch layout.
  // Capturing them against any other model would fail every bucket and the
  // negative cache would then pin those buckets to eager permanently --
  // skip warmup instead (lazy capture handles the model's real shapes).
  if (!HasRequestBatchLayout()) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("CUDA-graph warmup skipped for model instance '") +
         Name() +
         "': warmup widths are configured but the model does not expose the "
         "canonical request-batch layout")
            .c_str());
    return;
  }

  const int64_t single_width = model_state_->CudaGraphWarmupSingleWidth();
  const int64_t multi_width = model_state_->CudaGraphWarmupMultiWidth();
  const auto& buckets = model_state_->CudaGraphBatchSizes();

  // NOTE: warmup adds ~(num_buckets x ~150 ms) to this instance's load/READY
  // time.
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("CUDA-graph warmup: capturing ") +
       std::to_string(buckets.size()) +
       " bucket(s) at single_width=" + std::to_string(single_width) +
       " multi_width=" + std::to_string(multi_width) + " for model instance '" +
       Name() + "' (expect ~" + std::to_string(buckets.size() * 150) +
       " ms added to READY time)")
          .c_str());

  // One bucket attempt: zero-valued inputs at the bucket shape.
  const auto attempt_bucket = [&](int64_t bucket, const char* pass_name) {
    try {
      // request-batch layout at the bucket shape (see PadRequestsUp):
      // [0] packed_single (bucket, F1), [1] packed_multiple flat
      // (bucket*multi,), [2] request_end_position (bucket,) = cumsum of
      // per-request element counts.
      auto single = torch::zeros(
          {bucket, single_width},
          torch::dtype(torch::kFloat64).device(device_));
      auto multi = torch::zeros(
          {bucket * multi_width},
          torch::dtype(torch::kFloat64).device(device_));
      auto req_end =
          (torch::arange(
               1, bucket + 1, torch::dtype(torch::kInt64).device(device_)) *
           multi_width)
              .to(torch::kInt32);

      const auto start = std::chrono::steady_clock::now();
      const bool ok = CaptureBucket({single, multi, req_end}, bucket);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("CUDA-graph warmup bucket R=") + std::to_string(bucket) +
           (ok ? " captured in " : " FAILED after ") + std::to_string(ms) +
           " ms (" + pass_name + ") on model instance '" + Name() + "'")
              .c_str());
    }
    catch (const std::exception& ex) {
      // Never abort instance creation on a warmup failure -- lazy capture
      // covers it.
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("CUDA-graph warmup bucket R=") + std::to_string(bucket) +
           " threw (" + ex.what() +
           "); continuing (lazy capture is the safety net)")
              .c_str());
    }
  };

  // Capture LARGEST bucket first: it sizes the shared per-instance pool at
  // its maximum once, and every smaller bucket then fits inside existing
  // pool blocks (big blocks split down cleanly). Ascending order grew the
  // pool in steps and mid-sequence captures OOM-ed on fragmentation when
  // several instances packed one device.
  for (auto b_it = buckets.rbegin(); b_it != buckets.rend(); ++b_it) {
    attempt_bucket(*b_it, "first pass");
  }

  // Second chance for buckets that failed above: by now this instance's pool
  // is fully sized and the transient warmup allocations are returned
  // (CaptureBucket empties the allocator cache before capturing), so a
  // marginal-watermark OOM usually clears. One retry; a second failure
  // re-enters the negative cache and pins eager for real.
  std::vector<int64_t> retry_buckets;
  for (const int64_t bucket : buckets) {
    if (cuda_graph_failed_.count("R=" + std::to_string(bucket)) > 0) {
      retry_buckets.push_back(bucket);
    }
  }
  for (const int64_t bucket : retry_buckets) {
    cuda_graph_failed_.erase("R=" + std::to_string(bucket));
    attempt_bucket(bucket, "retry pass");
  }
}
#endif

TRITONSERVER_Error*
ModelInstanceState::GetNamingConvention(
    NamingConvention* naming_convention,
    const std::vector<std::string>& allowed_ios)
{
  // Rules for input/output tensor names with AOTInductor:
  // 1. Must follow the naming convention i.e. <name>__<index>
  // 2. If not, we enforce strict ordering from model configuration.
  //
  // Note: FORWARD_ARGUMENT convention is not available for AOTInductor models
  // since the model schema is not accessible at runtime.
  std::string deliminator = "__";
  std::string io_kind = allowed_ios.empty() ? "output" : "input";
  *naming_convention = NamingConvention::NAMED_INDEX;

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(
      model_state_->ModelConfig().MemberAsArray(io_kind.c_str(), &ios));

  // Check if inputs/outputs follow the <name>__<index> naming convention
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Validate name
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    int start_pos = io_name.find(deliminator);
    if (start_pos == -1) {
      *naming_convention = NamingConvention::STRICT_CONFIG_ORDERING;
      break;
    } else {
      // check if the index part of the name is not an integer
      std::string index_str = io_name.substr(start_pos + 2);
      bool is_int = true;
      for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
        if (std::isdigit(*itr) == 0) {
          is_int = false;
        }
      }

      if (!is_int) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            ((io_kind == "input" ? "input '" : "output '") + io_name +
             "' does not follow the <name>__<index> naming convention. "
             "Falling back to enforcing strict ordering from model "
             "configuration.")
                .c_str());
        *naming_convention = NamingConvention::STRICT_CONFIG_ORDERING;
        break;
      }
    }
  }

  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    // If we need to manage state for the model, then we need to check
    // the naming of the state adheres to both the input and output conventions
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      if (*naming_convention != NamingConvention::NAMED_INDEX) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("PyTorch model '" + model_state_->Name() +
             "' is using sequence batching with state but not all inputs and "
             "outputs follow the <name>__<index> naming convention. ")
                .c_str());
      }
    }

    for (size_t i = 0; i < states.ArraySize(); i++) {
      triton::common::TritonJson::Value state;
      RETURN_IF_ERROR(states.IndexAsObject(i, &state));
      std::string name_entry =
          io_kind == "input" ? "input_name" : "output_name";
      std::string state_name;
      RETURN_IF_ERROR(state.MemberAsString(name_entry.c_str(), &state_name));
      int start_pos = state_name.find(deliminator);
      if (start_pos == -1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("PyTorch model '" + model_state_->Name() +
             "' is using sequence batching with state but state '" +
             state_name +
             "' does not follow the <name>__<index> naming convention. ")
                .c_str());
      } else {
        // check if the index part of the name is not an integer
        std::string index_str = state_name.substr(start_pos + 2);
        bool is_int = true;
        for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
          if (std::isdigit(*itr) == 0) {
            is_int = false;
          }
        }
        if (!is_int) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              ("PyTorch model '" + model_state_->Name() +
               "' is using sequence batching with state but state '" +
               state_name +
               "' does not follow the <name>__<index> naming convention. ")
                  .c_str());
        }
      }
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());


#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    SetCurrentCudaStream(stream_, DeviceId());
  } else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
    // Replace the default stream of each device with the one we created.
    for (size_t i = 0; i < stream_vec_.size(); i++) {
      SetCurrentCudaStream(stream_vec_[i], i);
    }
  }
#endif

  NVTX_RANGE(nvtx_, "ProcessRequests " + Name());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to PyTorch backend for '" + Name() + "'")
                  .c_str()));
      return;
    }
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (size_t i = 0; i < request_count; i++) {
    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size.
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed, err);
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if (!all_response_failed) {
    if ((total_batch_size != 1) &&
        (total_batch_size > (size_t)max_batch_size)) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "batch size " + std::to_string(total_batch_size) + " for '" +
                  Name() + "', max allowed is " +
                  std::to_string(max_batch_size))
                  .c_str()));
    }
  }

  std::vector<const char*> input_names;
  std::vector<torch::Tensor> input_tensors;
  bool cuda_copy = false;
  std::unique_ptr<BackendInputCollector> collector;

  // For 'KIND_MODEL', it's fine to use CUDA events to calculate the compute
  // input duration since only one stream will be used for input collection.
  if ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) ||
      ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0))) {
#ifdef TRITON_ENABLE_GPU
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(
                compute_input_start_event_, GetCudaStreamByInstanceKind()),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  }

#ifdef TRITON_ENABLE_GPU
  // Input prestaging: when the bucket this batch would replay at is already
  // captured, SetInputTensors collects the payloads directly into its static
  // input buffers, and ExecuteWithCudaGraph skips PadRequestsUp + the
  // replay-time copy_ pass. Viability (widths/dtypes/uniformity) is
  // re-checked in SetInputTensors before any payload is consumed.
  prestaged_entry_ = nullptr;
  prestaged_bucket_ = -1;
  if (!all_response_failed && model_state_->EnabledCudaGraph() &&
      !device_.is_cpu()) {
    prestaged_entry_ = FindReadyCudaGraphEntry(
        static_cast<int64_t>(total_batch_size), &prestaged_bucket_);
  }
#endif

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
        GetCudaStreamByInstanceKind(), nullptr, nullptr, 0,
        HostPolicyName().c_str()));
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        SetInputTensors(
            total_batch_size, requests, request_count, &responses,
            collector.get(), &input_names, &input_tensors, &cuda_copy));
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(GetCudaStreamByInstanceKind());
    cuda_copy = false;
  }
#endif

  std::vector<torch::Tensor> output_tensors;
  uint64_t compute_start_ns = 0;
  uint64_t compute_infer_start = 0;

  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      RecordBackendTimestamp(
          &compute_start_ns,
          reinterpret_cast<void*>(&compute_infer_start_event_)));

  // For 'KIND_MODEL', capture the timestamp for the compute infer duration.
  if ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0)) {
    SET_TIMESTAMP(compute_infer_start);
  }

  // Run...
  if (!all_response_failed) {
    Execute(&responses, request_count, &input_tensors, &output_tensors);
  }

  // Verify output indices are valid with number of outputs after execution
  bool invalid_index = false;
  int max_index = output_tensors.size() - 1;

  if (!all_response_failed) {
    for (const auto& name : model_state_->ModelOutputs()) {
      int op_index = output_index_map_[name.first];
      if ((op_index < 0) || (op_index > max_index)) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "The output " + std::string(name.first) +
                    " in the model configuration refers to an output index "
                    "which doesn't exist. This model has " +
                    std::to_string(max_index + 1) + " outputs")
                    .c_str()));
        invalid_index = true;
        break;
      }
    }
  }

#ifdef TRITON_ENABLE_GPU
  // Note: Synchronization is already done in Execute() after model run.
  // For KIND_MODEL, if the model uses multiple devices internally, those
  // synchronizations happen within the model execution.
  // No additional synchronization needed here since Execute() already synced.
#endif

  uint64_t compute_end_ns = 0;
  uint64_t compute_output_start = 0;

  if ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0)) {
#ifdef TRITON_ENABLE_GPU
    SET_TIMESTAMP(compute_output_start);
#endif
  } else {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        RecordBackendTimestamp(
            &compute_end_ns,
            reinterpret_cast<void*>(&compute_output_start_event_)));
  }

  if (!all_response_failed) {
    if (!invalid_index) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          ReadOutputTensors(
              total_batch_size, output_tensors, requests, request_count,
              &responses));
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send PyTorch backend response");
    }
  }

  // We don't need an explicit CUDA synchronization here since we have already
  // synchronized the stream in the ReadOutputTensors function.
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    float compute_input_duration = GetCudaEventElapsedTime(
        compute_input_start_event_, compute_infer_start_event_);
    float compute_infer_duration = GetCudaEventElapsedTime(
        compute_infer_start_event_, compute_output_start_event_);

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + (compute_infer_duration * 1e6);
#endif
  } else if (
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0)) {
#ifdef TRITON_ENABLE_GPU
    float compute_input_duration = GetCudaEventElapsedTime(
        compute_input_start_event_, compute_infer_start_event_);
    uint64_t compute_infer_duration =
        compute_output_start - compute_infer_start;

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + compute_infer_duration;
#endif
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), total_batch_size, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<torch::Tensor>& output_tensors,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  // Thin containment wrapper -- same rationale as SetInputTensors: this runs
  // outside Execute's try/catch and must not let torch/responder exceptions
  // escape to the extern "C" boundary.
  try {
    return ReadOutputTensorsImpl(
        total_batch_size, output_tensors, requests, request_count, responses);
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("output processing failed: ") + ex.what()).c_str());
  }
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensorsImpl(
    size_t total_batch_size, const std::vector<torch::Tensor>& output_tensors,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  NVTX_RANGE(nvtx_, "ReadOutputTensors " + Name());

  bool use_pinned_input = model_state_->EnablePinnedInput();
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, use_pinned_input,
      GetCudaStreamByInstanceKind());

  bool cuda_copy = false;

  for (auto& output : model_state_->ModelOutputs()) {
    int op_index = output_index_map_[output.first];
    auto name = output.first;
    auto output_tensor_pair = output.second;

    torch::Tensor output_flat;
    try {
      output_flat = output_tensors[op_index].contiguous().flatten();
    }
    catch (std::exception& ex) {
      RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, (std::string("output tensor '") + name +
                                        "' is not found: " + ex.what())
                                           .c_str()));
    }

    // Verify output datatype matches datatype from model config
    TRITONSERVER_DataType output_dtype =
        ConvertTorchTypeToDataType(output_flat.scalar_type());
    TRITONSERVER_DataType config_datatype = output_dtype_map_[name];
    if (config_datatype != output_dtype) {
      RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("configuration expects datatype TYPE_") +
           TRITONSERVER_DataTypeString(config_datatype) + " for output '" +
           name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(output_dtype))
              .c_str()));
    }

    const char* output_buffer =
        static_cast<const char*>(output_flat.data_ptr());

    // Output tensors may not reside on the same device as model
    torch::Device tensor_device = output_flat.device();
    const auto memory_type = (tensor_device.type() == torch::kCPU)
                                 ? TRITONSERVER_MEMORY_CPU
                                 : TRITONSERVER_MEMORY_GPU;
    const auto memory_id =
        (tensor_device.type() == torch::kCPU) ? 0 : tensor_device.index();

    // Batch output doesn't support string data type yet, as it is not trivial
    // to parse string output
    const BatchOutput* batch_output = StateForModel()->FindBatchOutput(name);
    if (batch_output == nullptr) {
      // Get output shape
      std::vector<int64_t> batchn_shape;
      auto shape = output_tensors[op_index].sizes();
      for (auto itr = shape.begin(); itr != shape.end(); itr++) {
        batchn_shape.push_back(*itr);
      }

      if (batchn_shape.size() == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("output '") + name +
             "' is a scalar which is not supported.")
                .c_str());
      }
      if (output_tensor_pair.first != -1) {
        responder.ProcessTensor(
            name, output_dtype, batchn_shape, output_buffer, memory_type,
            memory_id);
      }
      if (output_tensor_pair.second != -1) {
        std::vector<TRITONBACKEND_State*> states;
        states = responder.ProcessStateTensor(
            name, output_dtype, batchn_shape, output_buffer, memory_type,
            memory_id);

        // Update the states
        for (auto& state : states) {
          RETURN_IF_ERROR(TRITONBACKEND_StateUpdate(state));
        }
      }

    } else {
      responder.ProcessBatchOutput(
          name, *batch_output, output_buffer, memory_type, memory_id);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  // Note: responder.Finalize() returns true if there are ANY pending CUDA
  // stream operations, not just copies. This includes pinned memory operations,
  // buffer preparation, etc. Even if outputs are on GPU, Finalize() may
  // return true due to internal Triton operations.
  //
  // Since we already synchronized in Execute() after model execution, outputs
  // should be ready. However, if Finalize() queued operations, we need to sync.
  // We use a non-blocking check first to see if sync is actually needed.
  if (cuda_copy) {
    cudaError_t status = cudaStreamQuery(GetCudaStreamByInstanceKind());
    if (status == cudaErrorNotReady) {
      // Stream has pending operations from responder, sync is needed
      cudaStreamSynchronize(GetCudaStreamByInstanceKind());
    }
  }
#endif

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::RecordBackendTimestamp(
    uint64_t* timestamp, void* cuda_event)
{
  if ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) ||
      ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0))) {
#ifdef TRITON_ENABLE_GPU
    cudaEvent_t* lcuda_event = reinterpret_cast<cudaEvent_t*>(cuda_event);
    RETURN_IF_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventRecord(*lcuda_event, GetCudaStreamByInstanceKind()),
        TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  } else {
    SET_TIMESTAMP(*timestamp);
  }
  return nullptr;
}

void
ModelInstanceState::SetCurrentCudaStream(
    const cudaStream_t& stream, const int& device_id)
{
#ifdef TRITON_ENABLE_GPU
  at::cuda::CUDAStream torch_stream =
      at::cuda::getStreamFromExternal(stream, device_id);
  // This function replaces the default stream with the stream we created. It
  // is not necessary to change the current device to the desired device when
  // replacing the default stream for that device. See the documentation here:
  // https://pytorch.org/cppdocs/api/function_namespacec10_1_1cuda_1a6ed50cc0fc16cc7014d9c2f4c3bd098d.html
  at::cuda::setCurrentCUDAStream(torch_stream);
#endif
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<torch::Tensor>* input_tensors, bool* cuda_copy)
{
  // Thin containment wrapper: torch ops in the implementation throw C++
  // exceptions (allocation failures foremost), this runs OUTSIDE Execute's
  // try/catch, and an exception escaping ProcessRequests crosses the
  // extern "C" boundary -> std::terminate, killing the whole server (a live
  // serving crashloop was exactly this class). Contain everything as a
  // request error instead.
  try {
    return SetInputTensorsImpl(
        total_batch_size, requests, request_count, responses, collector,
        input_names, input_tensors, cuda_copy);
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("input collection failed: ") + ex.what()).c_str());
  }
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensorsImpl(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<torch::Tensor>* input_tensors, bool* cuda_copy)
{
  // InferenceMode should be used to guard all tensors operations
  torch::InferenceMode infer_guard(model_state_->EnabledInferenceMode());

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  input_tensors->resize(input_count + batch_input_count_);

  // The inputs must be in contiguous CPU/GPU memory.
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
  if (device_.is_cpu()) {
    alloc_perference = {
        {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};
  } else {
    alloc_perference = {{TRITONSERVER_MEMORY_GPU, device_.index()}};
  }

  // Pass 1: gather every declared input's name/dtype/batched shape WITHOUT
  // collecting (request payloads are consumable, so the prestage decision must
  // precede the first ProcessTensor call). For ragged inputs also track
  // whether every request contributes the same element count -- the prestaged
  // static buffers assume uniform per-request slots.
  struct DeclaredInput {
    const char* name = nullptr;
    TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
    std::vector<int64_t> batchn_shape;
    int64_t batchn_elements = 0;
    bool ragged_uniform = true;
  };
  std::vector<DeclaredInput> declared(input_count);
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    input_names->emplace_back(input_name);
    DeclaredInput& info = declared[input_idx];
    info.name = input_name;
    info.datatype = input_datatype;

    // AOTInductor does not support string inputs
    if (input_datatype == TRITONSERVER_TYPE_BYTES) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("AOTInductor models do not support string/bytes input "
                       "type for input '") +
           input_name + "'")
              .c_str());
    }

    // The shape for the entire input patch,
    // [total_batch_size, ...] for non-ragged input and
    // [total_element_count] for ragged input (non-nested tensor)
    if (StateForModel()->IsInputRagged(input_name)) {
      info.batchn_shape = std::vector<int64_t>{0};
      int64_t first_element_cnt = -1;
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_InputProperties(
                                      input, nullptr, nullptr, &input_shape,
                                      &input_dims_count, nullptr, nullptr));

        int64_t element_cnt = 0;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            GetElementCount(input_shape, input_dims_count, &element_cnt));
        if (first_element_cnt < 0) {
          first_element_cnt = element_cnt;
        } else if (element_cnt != first_element_cnt) {
          info.ragged_uniform = false;
        }
        info.batchn_shape[0] += element_cnt;
#ifdef TRITON_ENABLE_GPU
        // A ragged request with NO elements cannot carry a real candidate
        // row. The canonical-layout model derives per-request row counts from
        // this input; running it on an empty request trips a device-side
        // assert INSIDE the model, which poisons the CUDA context and bricks
        // the instance (the server keeps answering, every inference fails) --
        // observed live with an empty packed_multiple probe, on the EAGER
        // path. Reject the batch up front instead. Deliberate blast radius:
        // EVERY request in this batch gets this INVALID_ARG (per-request
        // dropping mid-collection is not supported by the backend utils);
        // the batchmates' error is retryable, a bricked instance is not.
        if (element_cnt <= 0 && !device_.is_cpu() && HasRequestBatchLayout()) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("ragged input '") + input_name + "' request " +
               std::to_string(idx) +
               " is empty; the request-batch layout requires at least one "
               "row per request")
                  .c_str());
        }
#endif
      }
    } else {
      info.batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (supports_batching_) {
        info.batchn_shape[0] = total_batch_size;
      }
    }
    int64_t elements = 0;
    RETURN_IF_ERROR(GetElementCount(
        info.batchn_shape.data(),
        static_cast<uint32_t>(info.batchn_shape.size()), &elements));
    info.batchn_elements = elements;
  }

#ifdef TRITON_ENABLE_GPU
  // Input prestaging viability: every declared input must line up with the
  // resolved bucket entry's static buffer (dtype, per-request-slot width,
  // nonzero size, uniform ragged widths). Decided BEFORE any collection so the
  // batch is never split between paths; any mismatch falls back wholesale.
  if (prestaged_entry_ != nullptr) {
    const int64_t r = static_cast<int64_t>(total_batch_size);
    bool viable = (!device_.is_cpu()) && (r > 0) &&
                  (Kind() != TRITONSERVER_INSTANCEGROUPKIND_MODEL) &&
                  (prestaged_bucket_ >= r) &&
                  (prestaged_entry_->static_inputs.size() ==
                   static_cast<size_t>(input_count) + batch_input_count_);
    for (uint32_t i = 0; viable && (i < input_count); ++i) {
      const DeclaredInput& info = declared[i];
      const auto idx_it = input_index_map_.find(info.name);
      if (idx_it == input_index_map_.end() ||
          static_cast<size_t>(idx_it->second) >=
              prestaged_entry_->static_inputs.size()) {
        viable = false;
        break;
      }
      const torch::Tensor& st = prestaged_entry_->static_inputs[idx_it->second];
      const auto torch_dtype = ConvertDataTypeToTorchType(info.datatype);
      viable = st.is_cuda() && st.is_contiguous() &&
               (st.scalar_type() == torch_dtype.second) &&
               (st.numel() % prestaged_bucket_ == 0) &&
               (st.size(0) % prestaged_bucket_ == 0) &&
               (info.batchn_elements == r * (st.numel() / prestaged_bucket_)) &&
               (info.batchn_elements > 0) && info.ragged_uniform;
    }
    // Batch-input target buffers are skipped (content invariant per bucket)
    // but their r-sliced views use the same prefix math -- validate them too.
    if (viable) {
      for (const auto& batch_input : StateForModel()->BatchInputs()) {
        for (const auto& bi_name : batch_input.TargetNames()) {
          const auto idx_it = input_index_map_.find(bi_name);
          if (idx_it == input_index_map_.end() ||
              static_cast<size_t>(idx_it->second) >=
                  prestaged_entry_->static_inputs.size() ||
              !prestaged_entry_->static_inputs[idx_it->second].is_cuda() ||
              (prestaged_entry_->static_inputs[idx_it->second].size(0) %
                   prestaged_bucket_ !=
               0)) {
            viable = false;
            break;
          }
        }
        if (!viable) {
          break;
        }
      }
    }
    if (!viable) {
      prestaged_entry_ = nullptr;
      prestaged_bucket_ = -1;
    }
  }
#endif

  // Pass 2: collect. Prestaged batches are written by the collector DIRECTLY
  // into the graph entry's static input buffers (fixed replay addresses) --
  // the real requests occupy the first r slots and the padded tail is zeroed
  // here, replacing PadRequestsUp + the replay-time copy_ pass entirely.
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    const DeclaredInput& info = declared[input_idx];
    const std::vector<int64_t>& batchn_shape = info.batchn_shape;
    const auto torch_dtype = ConvertDataTypeToTorchType(info.datatype);

#ifdef TRITON_ENABLE_GPU
    if (prestaged_entry_ != nullptr) {
      const int64_t r = static_cast<int64_t>(total_batch_size);
      torch::Tensor& st =
          prestaged_entry_->static_inputs[input_index_map_[info.name]];
      const size_t batchn_byte_size =
          static_cast<size_t>(info.batchn_elements * st.element_size());
      // This ProcessTensor overload returns void; per-request failures are
      // reported through `responses` internally.
      collector->ProcessTensor(
          info.name, static_cast<char*>(st.data_ptr()), batchn_byte_size,
          TRITONSERVER_MEMORY_GPU, device_.index());
      const int64_t prefix_dim0 = r * (st.size(0) / prestaged_bucket_);
      if (prefix_dim0 < st.size(0)) {
        // Zero the padded tail (also clears stale rows from a previous,
        // larger batch). On the instance stream so the pre-replay event in
        // ReplayCudaGraphEntry orders it before the graph reads.
        const c10::cuda::CUDAStream prev_stream =
            c10::cuda::getCurrentCUDAStream(device_.index());
        c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromExternal(
            GetCudaStreamByInstanceKind(), device_.index()));
        st.narrow(0, prefix_dim0, st.size(0) - prefix_dim0).zero_();
        c10::cuda::setCurrentCUDAStream(prev_stream);
      }
      // Expose the r-sized view for shape derivation and eager fallback.
      (*input_tensors)[input_index_map_[info.name]] =
          st.narrow(0, 0, prefix_dim0);
      continue;
    }
#endif

    // The input must be in contiguous CPU/GPU memory.
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    // For 'KIND_MODEL', input will always be in CPU as we don't have a way to
    // query the input types.
    if (device_.is_cpu() || (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL)) {
      alloc_perference = {
          {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};
    } else {
      alloc_perference = {{TRITONSERVER_MEMORY_GPU, device_.index()}};
    }

    const char* input_buffer;
    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    RETURN_IF_ERROR(collector->ProcessTensor(
        info.name, nullptr, 0, alloc_perference, &input_buffer,
        &batchn_byte_size, &memory_type, &memory_type_id));

    // Create Torch tensor
    torch::TensorOptions options{torch_dtype.second};
    auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
                               ? options.device(torch::kCUDA, device_.index())
                               : options.device(torch::kCPU);

    if (batchn_byte_size) {
      // Remove constness to align with the signature of torch::from_blob()
      torch::Tensor input_tensor = torch::from_blob(
          const_cast<char*>(input_buffer), batchn_shape, updated_options);
      (*input_tensors)[input_index_map_[info.name]] = input_tensor;
    } else {
      // torch:from_blob seems not working when the input size is 0
      // create zero-length inputs directly
      torch::Tensor input_tensor = torch::zeros(batchn_shape, updated_options);
      (*input_tensors)[input_index_map_[info.name]] = input_tensor;
    }
  }

  for (const auto& batch_input : StateForModel()->BatchInputs()) {
    std::vector<int64_t> shape;
    collector->BatchInputShape(batch_input, &shape);

    for (const auto& input_name : batch_input.TargetNames()) {
      input_names->emplace_back(input_name.c_str());

#ifdef TRITON_ENABLE_GPU
      if (prestaged_entry_ != nullptr) {
        // Prestaged replay reads the captured static buffer directly, and a
        // bucket's batch-input content is invariant across batches (uniform
        // per-request widths make the accumulated-count values identical to
        // what capture recorded). Skip collection; expose an r-sliced view
        // for shape derivation and eager fallback.
        const auto idx = input_index_map_[input_name];
        torch::Tensor& st = prestaged_entry_->static_inputs[idx];
        const int64_t prefix_dim0 = static_cast<int64_t>(total_batch_size) *
                                    (st.size(0) / prestaged_bucket_);
        (*input_tensors)[idx] = st.narrow(0, 0, prefix_dim0);
        continue;
      }
#endif

      const char* dst_buffer;
      size_t dst_buffer_byte_size;
      TRITONSERVER_MemoryType dst_memory_type;
      int64_t dst_memory_type_id;

      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
          (*responses), responses->size(),
          collector->ProcessBatchInput(
              batch_input, nullptr, 0, alloc_perference, &dst_buffer,
              &dst_buffer_byte_size, &dst_memory_type, &dst_memory_type_id));

      const auto torch_dtype =
          ConvertDataTypeToTorchType(batch_input.DataType());
      torch::TensorOptions options{torch_dtype.second};
      auto updated_options = (dst_memory_type == TRITONSERVER_MEMORY_GPU)
                                 ? options.device(torch::kCUDA, device_.index())
                                 : options.device(torch::kCPU);

      if (dst_buffer_byte_size) {
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<char*>(dst_buffer), shape, updated_options);
        (*input_tensors)[input_index_map_[input_name]] = input_tensor;
      } else {
        // special handle when input has zero size
        torch::Tensor input_tensor = torch::zeros(shape, updated_options);
        (*input_tensors)[input_index_map_[input_name]] = input_tensor;
      }
    }
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();

  return nullptr;
}

ModelState*
ModelInstanceState::StateForModel() const
{
  return model_state_;
}

TRITONSERVER_Error*
ModelInstanceState::ValidateBooleanSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    std::string deliminator = "__";
    int ip_index = 0;
    int start_pos = tensor_name.find(deliminator);
    if (start_pos == -1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("input '" + tensor_name +
           "' does not follow <name>__<index> naming convention.")
              .c_str());
    }

    // check if the index part of the name is not an integer
    std::string index_str = tensor_name.substr(start_pos + 2);
    for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
      if (std::isdigit(*itr) == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("input '" + tensor_name +
             "' does not follow <name>__<index> naming convention.")
                .c_str());
      }
    }

    ip_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
    input_index_map_[tensor_name] = ip_index;
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
  // For AOTInductor models, we cannot inspect the model schema at runtime.
  // We rely on the model configuration to define inputs and use strict
  // ordering or named index convention.

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));

  if (ios.ArraySize() == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "model configuration must contain at least one input, none were "
        "specified.");
  }

  // For AOTInductor, use empty allowed_inputs (triggers NAMED_INDEX or
  // STRICT_CONFIG_ORDERING)
  std::vector<std::string> allowed_inputs;
  NamingConvention naming_convention;
  RETURN_IF_ERROR(GetNamingConvention(&naming_convention, allowed_inputs));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Validate name
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    AddInputToMap(naming_convention, io_name, i);

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTorchType(io_dtype);

    // AOTInductor does not support string/bytes type
    if (io_dtype == "TYPE_STRING") {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("AOTInductor models do not support string/bytes datatype for "
           "input '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    }

    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for input '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }
  }

  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string state_name;
        RETURN_IF_ERROR(state.MemberAsString("input_name", &state_name));
        AddInputToMap(naming_convention, state_name, i);

        // Validate data type
        std::string state_dtype;
        RETURN_IF_ERROR(state.MemberAsString("data_type", &state_dtype));
        const auto pr = ModelConfigDataTypeToTorchType(state_dtype);

        // AOTInductor does not support string/bytes type
        if (state_dtype == "TYPE_STRING") {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              ("AOTInductor models do not support string/bytes datatype for "
               "input state '" +
               state_name + "' for model '" + model_state_->Name() + "'")
                  .c_str());
        }

        if (!pr.first) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              ("unsupported datatype " + state_dtype + " for input state '" +
               state_name + "' for model '" + model_state_->Name() + "'")
                  .c_str());
        }
      }
    }
  }

  triton::common::TritonJson::Value batch_inputs;
  RETURN_IF_ERROR(
      model_state_->ModelConfig().MemberAsArray("batch_input", &batch_inputs));
  size_t i = 0;
  for (const auto& batch_input : StateForModel()->BatchInputs()) {
    for (const auto& input_name : batch_input.TargetNames()) {
      AddInputToMap(naming_convention, input_name, i + ios.ArraySize());
      i++;
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));
  std::string deliminator = "__";
  int op_index = 0;

  if (ios.ArraySize() == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "model configuration must contain at least one output, none were "
        "specified.");
  }

  NamingConvention naming_convention;
  RETURN_IF_ERROR(GetNamingConvention(&naming_convention, {}));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Validate name
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    switch (naming_convention) {
      case NamingConvention::NAMED_INDEX: {
        int start_pos = io_name.find(deliminator);
        op_index = std::atoi(io_name.substr(start_pos + 2).c_str());
        break;
      }
      case NamingConvention::FORWARD_ARGUMENT:
      case NamingConvention::STRICT_CONFIG_ORDERING: {
        op_index = i;
        break;
      }
    }

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTorchType(io_dtype);

    // AOTInductor does not support string/bytes type
    if (io_dtype == "TYPE_STRING") {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("AOTInductor models do not support string/bytes datatype for "
           "output '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    }

    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for output '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }

    output_index_map_[io_name] = op_index;
    output_dtype_map_[io_name] = ConvertTorchTypeToDataType(pr.second);
  }

  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string state_name;
        RETURN_IF_ERROR(state.MemberAsString("output_name", &state_name));
        std::string state_dtype;
        RETURN_IF_ERROR(state.MemberAsString("data_type", &state_dtype));
        std::vector<int64_t> dims;
        RETURN_IF_ERROR(ParseShape(state, "dims", &dims));

        // For state, naming convention is enforced to be NAMED_INDEX
        int start_pos = state_name.find(deliminator);
        op_index = std::atoi(state_name.substr(start_pos + 2).c_str());

        const auto pr = ModelConfigDataTypeToTorchType(state_dtype);

        // AOTInductor does not support string/bytes type
        if (state_dtype == "TYPE_STRING") {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              ("AOTInductor models do not support string/bytes datatype for "
               "output state '" +
               state_name + "' for model '" + model_state_->Name() + "'")
                  .c_str());
        }

        if (!pr.first) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              ("unsupported datatype " + state_dtype + " for state '" +
               state_name + "' for model '" + model_state_->Name() + "'")
                  .c_str());
        }

        output_index_map_[state_name] = op_index;
        output_dtype_map_[state_name] = ConvertTorchTypeToDataType(pr.second);
      }
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateTypedSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    std::string deliminator = "__";
    int ip_index = 0;
    int start_pos = tensor_name.find(deliminator);
    if (start_pos == -1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("input '" + tensor_name +
           "' does not follow <name>__<index> naming convention.")
              .c_str());
    }

    // check if the index part of the name is not an integer
    std::string index_str = tensor_name.substr(start_pos + 2);
    for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
      if (std::isdigit(*itr) == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("input '" + tensor_name +
             "' does not follow <name>__<index> naming convention.")
                .c_str());
      }
    }

    // check if the data type is supported by PyTorch
    if (!ModelConfigDataTypeToTorchType(tensor_datatype).first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("input '" + tensor_name + "' type '" + tensor_datatype +
           "' is not supported by PyTorch.")
              .c_str());
    }

    ip_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
    input_index_map_[tensor_name] = ip_index;
  }

  return nullptr;  // success
}


}  // namespace triton::backend::pytorch
