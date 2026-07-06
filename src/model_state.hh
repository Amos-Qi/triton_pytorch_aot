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
#include <mutex>
#include <set>

#include "libtorch_utils.h"
#include "naming_convention.hh"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/nvtx.h"
#include "triton/core/tritonbackend.h"

// for thread control
// https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#runtime-api
// https://github.com/pytorch/pytorch/blob/v2.2.1-rc3/aten/src/ATen/Parallel.h#L133
#include <ATen/Parallel.h>


namespace triton::backend::pytorch {

class ModelState : public triton::backend::BackendModel {
 private:
  // Flag to indicate whether inference mode is enabled. Defaults to true.
  bool enable_inference_mode_;

  // Flag to indicate whether cudnn is enabled. Defaults to true.
  bool enable_cudnn_;

  // Flag to indicate whether cache cleaning after each run is enabled.
  // Defaults to false.
  bool enable_cache_cleaning_;

  // Flag to indicate whether weight sharing is enabled. Defaults to false.
  bool enable_weight_sharing_;

  // Flag to enable whole-forward CUDA-graph capture/replay of the AOTInductor
  // model (ENABLE_CUDA_GRAPH parameter). Requires a static-shape .pt2. When on,
  // the loader is built run_single_threaded (single runner) so capture is legal.
  // Defaults to false (unchanged eager-AOTI path).
  bool enable_cuda_graph_;

  // R (request-batch) buckets to capture whole-forward CUDA graphs for (CUDA_GRAPH_BATCH_SIZES).
  // A batch is padded UP to the nearest bucket >= its R, replayed, and sliced; R above the max bucket
  // (or an empty set) falls back to eager. Empty = capture any first-seen shape (unbounded).
  std::set<int64_t> cuda_graph_batch_sizes_;

  // Flag to disable pinned input memory. Defaults to false (pinned input enabled by default).
  bool disable_pinned_input_;

  // Total instance count from instance_group configuration.
  // Used to determine num_runners when weight sharing is enabled.
  size_t total_instance_count_;

  // Model mapping for shared AOTInductor model across all instances on the
  // same device. The key is a pair of isGPU and device index.
  std::map<
      std::pair<bool, int64_t>,
      std::shared_ptr<torch::inductor::AOTIModelPackageLoader>>
      aoti_models_;

  // model_outputs is a map that contains unique outputs that the model must
  // provide. The first pair is the model output index and the second is
  // the index in the model state, -1 is used if one is not required.
  // In the model configuration, the output in the state configuration
  // can have intersection with the outputs section of the model. If an output
  // is specified both in the output section and state section, it indicates
  // that the backend must return the output state to the client too.
  std::map<std::string, std::pair<int64_t, int64_t>> model_outputs_;

 public:
  virtual ~ModelState() = default;

  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  bool EnabledCacheCleaning();

  bool EnabledCudnn();

  bool EnabledInferenceMode();

  bool EnabledWeightSharing();

  // Whether whole-forward CUDA-graph capture/replay is enabled for this model.
  bool EnabledCudaGraph();

  // The R buckets to capture CUDA graphs at (empty => capture any first-seen shape).
  const std::set<int64_t>& CudaGraphBatchSizes() const { return cuda_graph_batch_sizes_; }

  // Check if pinned input is disabled
  bool IsPinnedInputDisabled() const;

  // Custom EnablePinnedInput implementation to respect disable_pinned_input_ flag
  // Note: This is not an override since BackendModel::EnablePinnedInput() is not virtual
  bool EnablePinnedInput() const;

  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name, const torch::Device device,
      std::string* model_path, const TRITONSERVER_InstanceGroupKind& kind,
      std::shared_ptr<torch::inductor::AOTIModelPackageLoader>* aoti_model);

  const std::map<std::string, std::pair<int64_t, int64_t>>& ModelOutputs();

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  TRITONSERVER_Error* AutoCompleteConfig();

  TRITONSERVER_Error* ParseParameters();
};

}  // namespace triton::backend::pytorch
