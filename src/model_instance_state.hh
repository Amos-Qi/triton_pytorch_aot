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

#pragma once

#include <stdint.h>

#include <cstdint>
#include <exception>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "libtorch_utils.h"
#include "model_state.hh"
#include "naming_convention.hh"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/nvtx.h"
#include "triton/core/tritonbackend.h"

#ifdef TRITON_ENABLE_GPU
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif


namespace triton::backend::pytorch {

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 private:
  ModelState* model_state_;

  // The full path to the AOTInductor model package file.
  std::string model_path_;

  std::shared_ptr<torch::inductor::AOTIModelPackageLoader> aoti_model_;

  // Partial-graph split (ModelState::PartialSplit()): the eager row-dynamic
  // TRUNK loaded from model_trunk.pt2; aoti_model_ then holds the
  // graph-captured FRONT. Null unless the split is enabled.
  std::string trunk_model_path_;
  std::shared_ptr<torch::inductor::AOTIModelPackageLoader> trunk_aoti_model_;

  // Capture-legal loader for CUDA-graph capture (see ModelState::LoadModel).
  // Under weight sharing this is the shared single-threaded companion loader
  // (constants are references to aoti_model_'s tensors); otherwise it aliases
  // aoti_model_. Only used inside CaptureBucket, under the model-level
  // capture mutex.
  std::shared_ptr<torch::inductor::AOTIModelPackageLoader> capture_aoti_model_;
  torch::Device device_;

  // Map from configuration name for an input to the index of
  // that input in the model.
  std::unordered_map<std::string, int> input_index_map_;
  uint32_t batch_input_count_ = 0;

  // Map from configuration name for an output to the index of
  // that output in the model.
  std::unordered_map<std::string, int> output_index_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;

  // If the model supports batching.
  bool supports_batching_;

  cudaEvent_t compute_input_start_event_;
  cudaEvent_t compute_infer_start_event_;
  cudaEvent_t compute_output_start_event_;

  // Persistent event used to order the CUDA-graph replay stream behind the
  // producer (input-collection) stream without a full-device sync. Created once
  // (KIND_GPU + ENABLE_CUDA_GRAPH) in the ctor and destroyed in the destructor.
  cudaEvent_t cuda_graph_input_ready_event_ = nullptr;

  // Mirror of the above for the output direction: orders the instance stream
  // (responder output copies + next batch's input collection) behind the graph
  // replay, replacing the host-blocking post-replay stream.synchronize(). The
  // host proceeds to response preparation while the replay drains; the
  // responder's Finalize gate in ReadOutputTensors confirms completion before
  // responses send.
  cudaEvent_t cuda_graph_output_ready_event_ = nullptr;

  // Store the cuda streams created for the 'KIND_MODEL' instance group.
  std::vector<cudaStream_t> stream_vec_;

  // The number of available devices.
  int device_cnt_;

#ifdef TRITON_ENABLE_GPU
  // One captured whole-forward CUDA graph + its fixed I/O buffers and dedicated
  // capture/replay stream, keyed by input-shape signature. Populated lazily on
  // the first request of each shape and replayed on subsequent matching
  // requests. Only used when ModelState::EnabledCudaGraph() is true (which
  // requires a static-shape .pt2 and a run_single_threaded loader).
  struct CudaGraphEntry {
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    std::vector<torch::Tensor> static_inputs;
    std::vector<torch::Tensor> static_outputs;
    c10::cuda::CUDAStream stream;
  };
  std::unordered_map<std::string, CudaGraphEntry> cuda_graph_cache_;

  // One shared capture memory pool per INSTANCE: within an instance only one
  // graph replays at a time (batches are serialized), so every bucket's
  // intermediates can live in the same pool -- memory ~= the largest bucket
  // instead of the sum over buckets. First capture creates the pool; the
  // followers pass its id to capture_begin. {0,0} = not created yet.
  at::cuda::MempoolId_t cuda_graph_mempool_{0, 0};

  // Negative cache: input-shape keys whose capture failed once. We go straight
  // to eager for these (no re-warmup + re-capture + graph leak on every
  // subsequent request of the same shape).
  std::unordered_set<std::string> cuda_graph_failed_;

  // One-time guard (see ExecuteWithCudaGraph): on the first real request,
  // compare the configured warmup widths vs the actual traffic widths and, on
  // mismatch, evict the wrong-shape warmup captures so lazy capture
  // re-populates at the real shape (the cache is keyed only by "R=<bucket>",
  // not by width).
  bool warmup_widths_checked_ = false;

  // Tri-state cache for HasV3RequestBatchLayout(): 0 = not checked yet,
  // 1 = the model exposes the canonical request-batch layout, -1 = it doesn't.
  int v3_layout_state_ = 0;

  // Input prestaging (per-batch state; Triton runs batches serially per
  // instance). When SetInputTensors finds an already-captured bucket whose
  // widths match the incoming batch, it collects the request payloads DIRECTLY
  // into that entry's static_inputs (H2D into the graph's fixed addresses) and
  // records the entry here; ExecuteWithCudaGraph then replays without the
  // PadRequestsUp + static copy_ passes. nullptr = normal (slow) path.
  CudaGraphEntry* prestaged_entry_ = nullptr;
  int64_t prestaged_bucket_ = -1;

  // Per-batch: the ragged input's per-request element counts are NOT uniform
  // (ragged wire format -- the preprocessor sent real rows without padding to
  // the candidate bucket). Guards two paths that assume uniform widths: the
  // warmup width self-heal (a ragged batch cannot invalidate warmup captures)
  // and the slow-path PadRequestsUp/capture (per-request width math would be
  // garbage on a ragged batch; such batches replay via prestage or run eager).
  bool batch_ragged_nonuniform_ = false;

  // Partial-graph split, per-batch: REAL candidate-row count per request,
  // derived in SetInputTensors pass 1 from the ragged multi input's
  // per-request element counts / PartialSplitMultiRowWidth(). Drives the
  // boundary gather and the trunk's real end positions.
  std::vector<int64_t> partial_split_row_counts_;
#endif

 public:
  virtual ~ModelInstanceState();

  // Clear CUDA cache
  void ClearCache();

  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const;

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  void AddInputToMap(
      NamingConvention naming_convention, const std::string& io_name,
      const uint32_t index);

  // Create CUDA events for statistics collection.
  void CreateCudaEvents(const int32_t& device_id);

  void Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count, std::vector<torch::Tensor>* input_tensors,
      std::vector<torch::Tensor>* output_tensors);

  // Get the elapsed time between two CUDA events.
  float GetCudaEventElapsedTime(
      const cudaEvent_t& start_event, const cudaEvent_t& end_event);

  // Get the appropriate CUDA stream for input and output handling based on
  // the instance group type.
  cudaStream_t GetCudaStreamByInstanceKind();

#ifdef TRITON_ENABLE_GPU
  // Whether the model exposes the canonical request-batch layout:
  // packed_single_batch_tensor / packed_multiple_batch_tensor /
  // request_end_position at input indices 0/1/2. PadRequestsUp (and the
  // warmup's synthetic zero inputs) rewrite input[2] as a per-request cumsum,
  // which is only meaningful for this layout -- any other model replays
  // exact-R shapes only and never pads. Cached after the first call
  // (input_index_map_ is final after ValidateInputs).
  bool HasV3RequestBatchLayout();

  // Build a stable key from the input tensor shapes for the CUDA-graph cache.
  std::string InputShapeKey(const std::vector<torch::Tensor>& inputs) const;

  // Pad the request-batch inputs from R=r up to R=bucket (dummy requests
  // appended), so the batch can replay the bucket's captured graph. Returns the
  // padded inputs; caller slices outputs back to r.
  std::vector<torch::Tensor> PadRequestsUp(
      const std::vector<torch::Tensor>& inputs, int64_t r,
      int64_t bucket) const;

  // Capture-on-first-use + replay of the AOTI model for these (static-shape)
  // inputs on a dedicated stream. Returns false if capture is not possible
  // (the caller then falls back to eager aoti_model_->run). On success appends
  // the cloned model outputs to output_tensors.
  bool ExecuteWithCudaGraph(
      std::vector<torch::Tensor>* input_tensors,
      std::vector<torch::Tensor>* output_tensors);

  // Capture the whole-forward AOTI graph for R=<bucket> at the given (already
  // bucket-shaped) inputs on a dedicated stream and emplace it into
  // cuda_graph_cache_ (keyed "R=<bucket>"). Returns false on failure: it
  // negative-caches the bucket in cuda_graph_failed_, restores the stream,
  // WARNs, and LEAKs the partial at::cuda::CUDAGraph (its dtor must not run).
  // Used by both load-time warmup and lazy capture.
  bool CaptureBucket(
      const std::vector<torch::Tensor>& inputs_at_bucket, int64_t bucket);

  // Load-time warmup: before the instance goes READY, capture every configured
  // R bucket at zero-valued inputs of the configured warmup widths. No-op
  // unless ENABLE_CUDA_GRAPH, both warmup widths, and a non-empty bucket set
  // are all configured. Never throws (a warmup failure is logged; lazy capture
  // covers it).
  void WarmupCudaGraphs();

  // Input prestaging lookup: the bucket the slow path WOULD select for R=r,
  // but only if its graph is already captured (never captures). Mirrors the
  // ExecuteWithCudaGraph selection exactly: first healthy bucket >= r; if that
  // bucket is not captured yet, returns nullptr so the slow path performs its
  // capture-on-first-use. bucket_out is set only on a hit.
  CudaGraphEntry* FindReadyCudaGraphEntry(int64_t r, int64_t* bucket_out);

  // Replay `e` for a real request count r at R=bucket. When copy_from is
  // non-null, the inputs are first copied into e.static_inputs (the slow
  // path); nullptr means the batch was prestaged directly into the static
  // buffers by SetInputTensors and the copy is skipped. Appends r-sliced
  // cloned outputs to output_tensors. Returns false on replay failure (logged
  // + eager-fallback metric recorded; caller falls back to eager).
  bool ReplayCudaGraphEntry(
      CudaGraphEntry& e, int64_t r, int64_t bucket,
      const std::vector<torch::Tensor>* copy_from,
      std::vector<torch::Tensor>* output_tensors,
      bool narrow_outputs = true);

  // Partial-graph split execution: FRONT (graph replay via prestage, or eager
  // on the padded layout) -> boundary gather to real rows -> eager TRUNK.
  // Appends the trunk's outputs (already at the real request count) to
  // output_tensors. Throws std::runtime_error on unrecoverable failures
  // (Execute's catch converts it into error responses).
  void ExecutePartialSplit(
      std::vector<torch::Tensor>* input_tensors,
      std::vector<torch::Tensor>* output_tensors);

  // Rebuild the padded (R x candidate-bucket) front layout from the collected
  // wire tensors for the eager-front path (uncaptured bucket / replay
  // failure). gather_idx maps real rows to their padded positions.
  std::vector<torch::Tensor> BuildPaddedFrontInputs(
      const std::vector<torch::Tensor>& input_tensors, int64_t r,
      int64_t slot_rows, int64_t row_width, const torch::Tensor& gather_idx,
      size_t single_idx, size_t multi_idx, size_t end_idx);

  // Ragged-wire prestaging: place each request's real rows at its fixed slot
  // offset in the static buffer (slot i starts at i * slot_bytes) and zero the
  // per-slot tail, so an unpadded (ragged) wire batch lands in the exact
  // uniform bucket layout the graph was captured with. Copies + memsets are
  // issued async on the instance stream (the pre-replay event orders them).
  TRITONSERVER_Error* StageRaggedInputPerRequest(
      const char* input_name, TRITONBACKEND_Request** requests,
      const uint32_t request_count, torch::Tensor& dst, int64_t slot_elements);
#endif

  // Get the naming convention for inputs/outputs from the model configuration
  TRITONSERVER_Error* GetNamingConvention(
      NamingConvention* naming_convention,
      const std::vector<std::string>& allowed_io);

  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, const std::vector<torch::Tensor>& output_tensors,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  TRITONSERVER_Error* RecordBackendTimestamp(
      uint64_t* timestamp, void* cuda_event);

  // Replace the default CUDA stream with the stream we created to ensure
  // proper cuda stream synchronization.
  void SetCurrentCudaStream(
      const cudaStream_t& stream, const int32_t& device_id);

  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      std::vector<torch::Tensor>* input_tensors, bool* cuda_copy);

  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);

  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);

  TRITONSERVER_Error* ValidateOutputs();

  TRITONSERVER_Error* ValidateTypedSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
};

}  // namespace triton::backend::pytorch
