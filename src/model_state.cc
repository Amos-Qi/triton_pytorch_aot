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

#include "model_state.hh"

#include <mutex>

#include "cuda_graph_bucketing.h"


namespace {
std::once_flag pytorch_interop_threads_flag;
std::once_flag pytorch_intraop_threads_flag;
}  // namespace

namespace triton::backend::pytorch {

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), enable_inference_mode_(true),
      enable_cudnn_(true), enable_cache_cleaning_(false),
      enable_weight_sharing_(false), enable_cuda_graph_(false),
      disable_pinned_input_(false), total_instance_count_(1),
      cuda_graph_warmup_single_width_(0), cuda_graph_warmup_multi_width_(0),
      partial_split_(false), partial_split_multi_row_width_(0)
{
}

ModelState::~ModelState()
{
  // Metrics must be deleted before their families. Only non-null handles were
  // ever created (see InitCudaGraphMetrics / GetOrCreateBucketMetric).
  for (auto& entry : cuda_graph_bucket_metrics_) {
    if (entry.second != nullptr) {
      LOG_IF_ERROR(
          TRITONSERVER_MetricDelete(entry.second),
          "Failed to delete CUDA-graph bucket metric");
    }
  }
  if (metric_eager_fallbacks_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_MetricDelete(metric_eager_fallbacks_),
        "Failed to delete CUDA-graph eager-fallback metric");
  }
  if (metric_family_replays_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_MetricFamilyDelete(metric_family_replays_),
        "Failed to delete cudagraph_replays_total family");
  }
  if (metric_family_pad_waste_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_MetricFamilyDelete(metric_family_pad_waste_),
        "Failed to delete cudagraph_pad_waste_rows_total family");
  }
  if (metric_family_eager_fallbacks_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_MetricFamilyDelete(metric_family_eager_fallbacks_),
        "Failed to delete cudagraph_eager_fallbacks_total family");
  }
  if (metric_family_capture_failures_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_MetricFamilyDelete(metric_family_capture_failures_),
        "Failed to delete cudagraph_capture_failures_total family");
  }
}

void
ModelState::InitCudaGraphMetrics()
{
  metric_family_replays_ = CreateCudaGraphMetricFamily(
      "cudagraph_replays_total", "CUDA-graph replays (per R bucket)");
  metric_family_pad_waste_ = CreateCudaGraphMetricFamily(
      "cudagraph_pad_waste_rows_total",
      "Padded (dummy) request rows run through CUDA-graph replay (per R "
      "bucket)");
  metric_family_eager_fallbacks_ = CreateCudaGraphMetricFamily(
      "cudagraph_eager_fallbacks_total",
      "Requests that fell back to eager AOTInductor instead of a CUDA graph");
  metric_family_capture_failures_ = CreateCudaGraphMetricFamily(
      "cudagraph_capture_failures_total",
      "CUDA-graph capture failures (per R bucket)");

  // The eager-fallback counter is unlabeled -> one metric for the whole model.
  if (metric_family_eager_fallbacks_ != nullptr) {
    TRITONSERVER_Error* err = TRITONSERVER_MetricNew(
        &metric_eager_fallbacks_, metric_family_eager_fallbacks_,
        nullptr /* labels */, 0 /* label_count */);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      metric_eager_fallbacks_ = nullptr;
    }
  }
}

TRITONSERVER_MetricFamily*
ModelState::CreateCudaGraphMetricFamily(
    const char* name, const char* description)
{
  TRITONSERVER_MetricFamily* family = nullptr;
  TRITONSERVER_Error* err = TRITONSERVER_MetricFamilyNew(
      &family, TRITONSERVER_METRIC_KIND_COUNTER, name, description);
  if (err != nullptr) {
    if (!cuda_graph_metrics_warned_) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("Failed to create CUDA-graph metric family '") + name +
           "' (" + TRITONSERVER_ErrorMessage(err) +
           "); CUDA-graph metrics disabled for model '" + Name() + "'")
              .c_str());
      cuda_graph_metrics_warned_ = true;
    }
    TRITONSERVER_ErrorDelete(err);
    return nullptr;
  }
  return family;
}

TRITONSERVER_Metric*
ModelState::GetOrCreateBucketMetric(
    TRITONSERVER_MetricFamily* family, int64_t bucket)
{
  if (family == nullptr) {
    return nullptr;
  }
  // Instances execute (and record metrics) concurrently; the map must not be
  // read while another thread inserts.
  std::lock_guard<std::mutex> lk(cuda_graph_metrics_mutex_);
  const auto cache_key = std::make_pair(family, bucket);
  auto it = cuda_graph_bucket_metrics_.find(cache_key);
  if (it != cuda_graph_bucket_metrics_.end()) {
    return it->second;
  }

  TRITONSERVER_Metric* metric = nullptr;
  const std::string bucket_str = std::to_string(bucket);
  TRITONSERVER_Parameter* label = TRITONSERVER_ParameterNew(
      "bucket", TRITONSERVER_PARAMETER_STRING, bucket_str.c_str());
  if (label != nullptr) {
    const TRITONSERVER_Parameter* labels[1] = {label};
    TRITONSERVER_Error* err =
        TRITONSERVER_MetricNew(&metric, family, labels, 1);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      metric = nullptr;
    }
    TRITONSERVER_ParameterDelete(label);
  }
  // Cache even nullptr so a failed creation is not retried on every request.
  cuda_graph_bucket_metrics_[cache_key] = metric;
  return metric;
}

void
ModelState::CudaGraphMetricReplay(int64_t bucket)
{
  TRITONSERVER_Metric* metric =
      GetOrCreateBucketMetric(metric_family_replays_, bucket);
  if (metric != nullptr) {
    TRITONSERVER_Error* err = TRITONSERVER_MetricIncrement(metric, 1.0);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
ModelState::CudaGraphMetricPadWaste(int64_t bucket, int64_t rows)
{
  TRITONSERVER_Metric* metric =
      GetOrCreateBucketMetric(metric_family_pad_waste_, bucket);
  if (metric != nullptr) {
    TRITONSERVER_Error* err =
        TRITONSERVER_MetricIncrement(metric, static_cast<double>(rows));
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
ModelState::CudaGraphMetricEagerFallback()
{
  if (metric_eager_fallbacks_ != nullptr) {
    TRITONSERVER_Error* err =
        TRITONSERVER_MetricIncrement(metric_eager_fallbacks_, 1.0);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
ModelState::CudaGraphMetricCaptureFailure(int64_t bucket)
{
  TRITONSERVER_Metric* metric =
      GetOrCreateBucketMetric(metric_family_capture_failures_, bucket);
  if (metric != nullptr) {
    TRITONSERVER_Error* err = TRITONSERVER_MetricIncrement(metric, 1.0);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration is not supported since PyTorch does not
  // store/capture sufficient model metadata so just log error instead.
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for pytorch backend")
          .c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());
    RETURN_IF_ERROR((*state)->SetModelConfig());
  }

  auto& model_outputs = (*state)->model_outputs_;
  // Parse the output states in the model configuration
  triton::common::TritonJson::Value sequence_batching;
  if ((*state)->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string output_state_name;
        RETURN_IF_ERROR(
            state.MemberAsString("output_name", &output_state_name));
        auto it = model_outputs.find(output_state_name);
        if (it == model_outputs.end()) {
          model_outputs.insert({output_state_name, std::make_pair(-1, i)});
        } else {
          it->second.second = i;
        }
      }
    }
  }

  // Parse the output names in the model configuration
  triton::common::TritonJson::Value outputs;
  RETURN_IF_ERROR((*state)->ModelConfig().MemberAsArray("output", &outputs));
  for (size_t i = 0; i < outputs.ArraySize(); i++) {
    triton::common::TritonJson::Value output;
    THROW_IF_BACKEND_INSTANCE_ERROR(outputs.IndexAsObject(i, &output));

    // Use names from ModelConfig by reference since the model
    // config will persist longer than this inference execution.
    std::string output_name;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        output.MemberAsString("name", &output_name));

    auto it = model_outputs.find(output_name);
    if (it == model_outputs.end()) {
      model_outputs.insert({output_name, std::make_pair(i, -1)});
    } else {
      it->second.first = i;
    }
  }

  // Parse instance_group to get total instance count for weight sharing
  triton::common::TritonJson::Value instance_groups;
  if ((*state)->ModelConfig().Find("instance_group", &instance_groups)) {
    size_t total_count = 0;
    for (size_t i = 0; i < instance_groups.ArraySize(); i++) {
      triton::common::TritonJson::Value group;
      RETURN_IF_ERROR(instance_groups.IndexAsObject(i, &group));
      int64_t count = 1;
      group.MemberAsInt("count", &count);
      total_count += static_cast<size_t>(count);
    }
    if (total_count > 0) {
      (*state)->total_instance_count_ = total_count;
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Total instance count: ") + std::to_string(total_count) +
           " for model '" + (*state)->Name() + "'")
              .c_str());
    }
  }

  RETURN_IF_ERROR((*state)->ParseParameters());

  return nullptr;  // success
}

bool
ModelState::EnabledCacheCleaning()
{
  return enable_cache_cleaning_;
}

bool
ModelState::EnabledCudnn()
{
  return enable_cudnn_;
}

bool
ModelState::EnabledInferenceMode()
{
  return enable_inference_mode_;
}

bool
ModelState::EnabledWeightSharing()
{
  return enable_weight_sharing_;
}

bool
ModelState::EnabledCudaGraph()
{
  return enable_cuda_graph_;
}

bool
ModelState::IsPinnedInputDisabled() const
{
  return disable_pinned_input_;
}

bool
ModelState::EnablePinnedInput() const
{
  // If disable_pinned_input_ is true, return false to disable pinned input
  // Otherwise, use the default behavior from BackendModel
  if (disable_pinned_input_) {
    return false;
  }
  return BackendModel::EnablePinnedInput();
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name, const torch::Device device,
    std::string* model_path, const TRITONSERVER_InstanceGroupKind& kind,
    std::shared_ptr<torch::inductor::AOTIModelPackageLoader>* aoti_model,
    std::shared_ptr<torch::inductor::AOTIModelPackageLoader>* capture_model)
{
  // Find the AOTInductor package file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.pt2").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.pt2";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + *model_path +
            "' for model instance '" + Name() + "'");
  }

  // Weight sharing + CUDA graphs is the dual-loader mode: a thread-safe
  // multi-runner loader serves eager runs from every instance concurrently,
  // and a separate single-threaded loader exists ONLY for graph capture
  // (capture through the thread-safe run() path is illegal: its model-slot
  // reclamation queries completion events, which errors while the stream is
  // capturing -- pytorch/pytorch@85467ed). The capture loader's constants are
  // user-managed references to the serve loader's tensors, so the weights
  // exist ONCE and the captured graphs read the same addresses eager does.
  const bool dual_capture_mode =
      enable_weight_sharing_ && enable_cuda_graph_ && !device.is_cpu();

  // If weight sharing is enabled, skip loading model if it is already
  // available on the target device. Keyed by artifact too, so distinct
  // artifacts of one model (partial split) don't collide.
  std::pair<std::string, std::pair<bool, int64_t>> cache_key;
  if (enable_weight_sharing_) {
    cache_key = std::make_pair(
        cc_model_filename,
        std::make_pair(!device.is_cpu(), (int64_t)device.index()));
    auto mit = aoti_models_.find(cache_key);
    if (mit != aoti_models_.end()) {
      *aoti_model = mit->second;
      if (capture_model != nullptr) {
        auto cit = aoti_capture_models_.find(cache_key);
        if (cit != aoti_capture_models_.end()) {
          *capture_model = cit->second;
        } else if (dual_capture_mode) {
          // The multi-runner serve loader is capture-ILLEGAL (its model-slot
          // reclamation errors while a stream is capturing); handing it out
          // as the capture loader would fail every capture at runtime. The
          // companion is always cached with the serve loader on first load,
          // so this indicates a load-order bug -- fail loudly.
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("weight-sharing capture companion missing from "
                           "the loader cache for '") +
               Name() + "'")
                  .c_str());
        } else {
          *capture_model = *aoti_model;
        }
      }
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Reusing AOTInductor model for instance '") + Name() +
           "'")
              .c_str());
      return nullptr;  // success
    }
  }

  // InferenceMode should be used to guard all tensors operations including
  // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
  torch::InferenceMode infer_guard(EnabledInferenceMode());

  std::shared_ptr<torch::inductor::AOTIModelPackageLoader> capture_loader;
  try {
    // Determine the device index for AOTInductor
    int device_index = -1;  // Default for CPU or auto-selection
    if (!device.is_cpu()) {
      device_index = device.index();
    }

    // When weight sharing is enabled, use num_runners equal to the instance
    // count to allow concurrent inference from all instances sharing the model.
    size_t num_runners = enable_weight_sharing_ ? total_instance_count_ : 1;
    bool run_single_threaded = false;

    if (enable_cuda_graph_ && device.is_cpu()) {
      // CUDA graphs cannot exist on a CPU instance; the instance runs the
      // normal eager path. Don't degrade its loader to single-threaded.
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("ENABLE_CUDA_GRAPH is set but instance '") + Name() +
           "' runs on CPU; CUDA graphs are disabled for it (eager run, "
           "default threading)")
              .c_str());
    } else if (enable_cuda_graph_ && !dual_capture_mode) {
      // No sharing: the ONE loader doubles as the capture loader, so it must
      // be single-threaded (see the dual-mode comment above).
      run_single_threaded = true;
      num_runners = 1;
    }

    // Load the AOTInductor package
    aoti_model->reset(new torch::inductor::AOTIModelPackageLoader(
        *model_path, "model" /* model_name */, run_single_threaded, num_runners,
        device_index));

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Loaded AOTInductor model from '") + *model_path +
         "' with " + std::to_string(num_runners) + " runner(s)" +
         " for instance '" + Name() + "'")
            .c_str());

    if (dual_capture_mode && capture_model != nullptr) {
      // Second, capture-legal loader for the same artifact. Constructing it
      // loads its own constants first (a transient duplicate at load time);
      // load_constants(user_managed=true) then rebinds them to the serve
      // loader's tensors and releases the copies.
      capture_loader.reset(new torch::inductor::AOTIModelPackageLoader(
          *model_path, "model", /*run_single_threaded=*/true,
          /*num_runners=*/1, device_index));

      auto* serve_runner = (*aoti_model)->get_runner();
      const auto extracted =
          serve_runner->extract_constants_map(/*use_inactive=*/false);
      std::unordered_map<std::string, at::Tensor> shared_constants(
          extracted.begin(), extracted.end());
      // check_full_update=false: extract skips constants produced by constant
      // folding; the capture loader folds its own from the shared originals
      // on its first (warmup) run.
      capture_loader->load_constants(
          shared_constants, /*use_inactive=*/false,
          /*check_full_update=*/false, /*user_managed=*/true);

      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Weight-shared capture loader ready for '") +
           *model_path + "': " + std::to_string(shared_constants.size()) +
           " constants rebound to the serve loader's tensors for instance '" +
           Name() + "'")
              .c_str());
    }
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to load AOTInductor model '" + Name() + "': " + ex.what())
            .c_str());
  }

  if (capture_model != nullptr) {
    *capture_model = (capture_loader != nullptr) ? capture_loader : *aoti_model;
  }

  if (enable_weight_sharing_) {
    if (!((aoti_models_.emplace(cache_key, *aoti_model)).second)) {
      std::string type = device.is_cpu() ? "CPU" : "GPU";
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("Model already found on target ") + type + " device " +
           "(id " + std::to_string(device.index()) + ") for '" + Name() + "'")
              .c_str());
    }
    if (capture_loader != nullptr) {
      aoti_capture_models_.emplace(cache_key, capture_loader);
    }
  }

  return nullptr;  // success
}

const std::map<std::string, std::pair<int64_t, int64_t>>&
ModelState::ModelOutputs()
{
  return model_outputs_;
}

TRITONSERVER_Error*
ModelState::ParseParameters()
{
  triton::common::TritonJson::Value params;
  bool status = model_config_.Find("parameters", &params);
  if (status) {
    // If 'ENABLE_CACHE_CLEANING' is not present in 'parameters' then
    // no update is made to 'enable_cache_cleaning_'.
    TRITONSERVER_Error* err = ParseParameter(
        params, "ENABLE_CACHE_CLEANING", &enable_cache_cleaning_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Cache Cleaning is ") +
         (enable_cache_cleaning_ ? "enabled" : "disabled") +
         " for model instance '" + Name() + "'")
            .c_str());

    // If 'INFERENCE_MODE' is not present in 'parameters' then no update is made
    // to 'enable_inference_mode_'.
    err = ParseParameter(params, "INFERENCE_MODE", &enable_inference_mode_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Inference Mode is ") +
         (enable_inference_mode_ ? "enabled" : "disabled") +
         " for model instance '" + Name() + "'")
            .c_str());

    // If 'DISABLE_CUDNN' is not present in 'parameters' then no update is made
    // to 'enable_cudnn_'.
    bool disable_cudnn = false;
    err = ParseParameter(params, "DISABLE_CUDNN", &disable_cudnn);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
    enable_cudnn_ = !disable_cudnn;
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("cuDNN is ") + (enable_cudnn_ ? "enabled" : "disabled") +
         " for model instance '" + Name() + "'")
            .c_str());

    // If 'ENABLE_WEIGHT_SHARING' is not present in 'parameters' then no
    // update is made to 'enable_weight_sharing'.
    err = ParseParameter(
        params, "ENABLE_WEIGHT_SHARING", &enable_weight_sharing_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Weight sharing is ") +
           (enable_weight_sharing_ ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // If 'ENABLE_CUDA_GRAPH' is not present in 'parameters' then no update is
    // made to 'enable_cuda_graph_' (defaults to false).
    err = ParseParameter(params, "ENABLE_CUDA_GRAPH", &enable_cuda_graph_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("CUDA graph capture/replay is ") +
           (enable_cuda_graph_ ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // 'CUDA_GRAPH_BATCH_SIZES' (comma-separated R buckets) -> the allowlist the
    // backend captures graphs at + pads batches up to. Empty (param absent) =>
    // capture any first-seen shape (unbounded).
    {
      triton::common::TritonJson::Value cbs;
      if (params.Find("CUDA_GRAPH_BATCH_SIZES", &cbs)) {
        std::string val;
        TRITONSERVER_Error* serr = cbs.MemberAsString("string_value", &val);
        if (serr != nullptr) {
          TRITONSERVER_ErrorDelete(serr);
        } else {
          cuda_graph_batch_sizes_ = ParseCsvInt64Set(val);
          // Drop entries no real batch can ever match: non-positive values
          // (warmup would try to capture at them) and values above
          // max_batch_size (the scheduler never forms such batches; their
          // graphs would only waste capture time + VRAM).
          const int64_t max_bs = static_cast<int64_t>(MaxBatchSize());
          for (auto b_it = cuda_graph_batch_sizes_.begin();
               b_it != cuda_graph_batch_sizes_.end();) {
            if (*b_it <= 0 || (max_bs > 0 && *b_it > max_bs)) {
              LOG_MESSAGE(
                  TRITONSERVER_LOG_WARN,
                  (std::string("Ignoring CUDA_GRAPH_BATCH_SIZES entry ") +
                   std::to_string(*b_it) + " for model '" + Name() +
                   "' (must be in [1, max_batch_size=" +
                   std::to_string(max_bs) + "])")
                      .c_str());
              b_it = cuda_graph_batch_sizes_.erase(b_it);
            } else {
              ++b_it;
            }
          }
          LOG_MESSAGE(
              TRITONSERVER_LOG_INFO,
              (std::string("CUDA graph R buckets: '") + val +
               "' for model instance '" + Name() + "'")
                  .c_str());
        }
      }
    }

    // Weight sharing + CUDA graphs run in the dual-loader mode (see
    // LoadModel): a thread-safe multi-runner serve loader shared by all
    // instances plus one single-threaded capture loader whose constants are
    // user-managed references to the serve loader's tensors. Captures are
    // serialized by CudaGraphCaptureMutex(); graphs/replays stay per-instance.
    if (enable_cuda_graph_ && enable_weight_sharing_) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("ENABLE_CUDA_GRAPH + ENABLE_WEIGHT_SHARING: dual-loader "
                       "mode (shared weights, serialized captures) for model "
                       "instance '") +
           Name() + "'")
              .c_str());
    }

    // 'CUDA_GRAPH_WARMUP_SINGLE_WIDTH' / 'CUDA_GRAPH_WARMUP_MULTI_WIDTH'
    // (int64): packed_single's F1 width and the per-request packed_multiple
    // element count (candidate_bucket * F2). When both > 0 (with a non-empty
    // bucket set) the instance captures every bucket at load time (see
    // WarmupCudaGraphs). Absent
    // => 0 => lazy capture only.
    {
      triton::common::TritonJson::Value w;
      if (params.Find("CUDA_GRAPH_WARMUP_SINGLE_WIDTH", &w)) {
        std::string val;
        TRITONSERVER_Error* serr = w.MemberAsString("string_value", &val);
        if (serr != nullptr) {
          TRITONSERVER_ErrorDelete(serr);
        } else {
          try {
            cuda_graph_warmup_single_width_ = std::stoll(val);
          }
          catch (...) {
          }
          LOG_MESSAGE(
              TRITONSERVER_LOG_INFO,
              (std::string("CUDA graph warmup single width: ") +
               std::to_string(cuda_graph_warmup_single_width_) +
               " for model instance '" + Name() + "'")
                  .c_str());
        }
      }
    }
    {
      triton::common::TritonJson::Value w;
      if (params.Find("CUDA_GRAPH_WARMUP_MULTI_WIDTH", &w)) {
        std::string val;
        TRITONSERVER_Error* serr = w.MemberAsString("string_value", &val);
        if (serr != nullptr) {
          TRITONSERVER_ErrorDelete(serr);
        } else {
          try {
            cuda_graph_warmup_multi_width_ = std::stoll(val);
          }
          catch (...) {
          }
          LOG_MESSAGE(
              TRITONSERVER_LOG_INFO,
              (std::string("CUDA graph warmup multi width: ") +
               std::to_string(cuda_graph_warmup_multi_width_) +
               " for model instance '" + Name() + "'")
                  .c_str());
        }
      }
    }

    // 'PARTIAL_SPLIT' (bool): partial-graph split serving. model.pt2 is the
    // graph-captured FRONT (padded wire -> seam activations) and
    // model_trunk.pt2 is the row-dynamic TRUNK (real-row packed features +
    // boundary-gathered seam -> served heads), run eager after the front.
    err = ParseParameter(params, "PARTIAL_SPLIT", &partial_split_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Partial-graph split is ") +
           (partial_split_ ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // 'PARTIAL_SPLIT_MULTI_ROW_WIDTH' (int64): the per-candidate-row element
    // width (F2) of the ragged packed_multiple input. The boundary gather
    // needs it to convert the wire's per-request element counts into row
    // counts, and the eager-front path needs it to rebuild the padded layout.
    {
      triton::common::TritonJson::Value w;
      if (params.Find("PARTIAL_SPLIT_MULTI_ROW_WIDTH", &w)) {
        std::string val;
        TRITONSERVER_Error* serr = w.MemberAsString("string_value", &val);
        if (serr != nullptr) {
          TRITONSERVER_ErrorDelete(serr);
        } else {
          try {
            partial_split_multi_row_width_ = std::stoll(val);
          }
          catch (...) {
          }
          LOG_MESSAGE(
              TRITONSERVER_LOG_INFO,
              (std::string("Partial-split multi row width: ") +
               std::to_string(partial_split_multi_row_width_) +
               " for model instance '" + Name() + "'")
                  .c_str());
        }
      }
    }

    // The split path's gather/pad math is undefined without the row width and
    // the per-request slot width (candidate_bucket * F2, shipped as the warmup
    // multi width). Fail load loudly rather than serving wrong numbers.
    if (partial_split_) {
      if (partial_split_multi_row_width_ <= 0 ||
          cuda_graph_warmup_multi_width_ <= 0 ||
          (cuda_graph_warmup_multi_width_ % partial_split_multi_row_width_) !=
              0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("PARTIAL_SPLIT requires PARTIAL_SPLIT_MULTI_ROW_WIDTH "
                         "and CUDA_GRAPH_WARMUP_MULTI_WIDTH (a positive "
                         "multiple of the row width); got row_width=") +
             std::to_string(partial_split_multi_row_width_) +
             " multi_width=" + std::to_string(cuda_graph_warmup_multi_width_) +
             " for model '" + Name() + "'")
                .c_str());
      }
    }

    // If 'DISABLE_PINNED_INPUT' is not present in 'parameters' then no
    // update is made to 'disable_pinned_input_'.
    bool disable_pinned_input = false;
    err = ParseParameter(params, "DISABLE_PINNED_INPUT", &disable_pinned_input);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      disable_pinned_input_ = disable_pinned_input;
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Pinned input is ") +
           (disable_pinned_input_ ? "disabled" : "enabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // If 'INTRA_OP_THREAD_COUNT' is not present in 'parameters' then no update
    // is made to 'intra_op_thread_count', which by default will take all
    // threads
    int intra_op_thread_count = -1;
    err =
        ParseParameter(params, "INTRA_OP_THREAD_COUNT", &intra_op_thread_count);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      if (intra_op_thread_count > 0) {
        // at::set_num_threads() does not throw if called more than once, but
        // issues warnings. std::call_once() is useful to limit these.
        std::call_once(pytorch_intraop_threads_flag, [intra_op_thread_count]() {
          at::set_num_threads(intra_op_thread_count);
        });
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Intra op thread count is set to ") +
             std::to_string(at::get_num_threads()) + " for model instance '" +
             Name() + "'")
                .c_str());
      }
    }

    // If 'INTER_OP_THREAD_COUNT' is not present in 'parameters' then no update
    // is made to 'inter_op_thread_count', which by default will take all
    // threads
    int inter_op_thread_count = -1;
    err =
        ParseParameter(params, "INTER_OP_THREAD_COUNT", &inter_op_thread_count);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      if (inter_op_thread_count > 0) {
        // at::set_num_interop_threads() throws if called more than once.
        // std::call_once() should prevent this, but try/catch is additionally
        // used for safety.
        std::call_once(pytorch_interop_threads_flag, [inter_op_thread_count]() {
          try {
            at::set_num_interop_threads(inter_op_thread_count);
          }
          catch (const c10::Error& e) {
            // do nothing
          }
        });
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Inter op thread count is set to ") +
             std::to_string(at::get_num_interop_threads()) +
             " for model instance '" + Name() + "'")
                .c_str());
      }
    }
  }

  // Create the CUDA-graph Prometheus counters once (best-effort).
  // enable_cuda_graph_ is final by this point; the guard makes this a no-op for
  // the non-cudagraph path.
  if (enable_cuda_graph_) {
    InitCudaGraphMetrics();
  }

  return nullptr;
}

}  // namespace triton::backend::pytorch
