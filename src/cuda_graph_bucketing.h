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

#include <cstdint>
#include <set>
#include <string>

namespace triton::backend::pytorch {

// Pure, dependency-free helpers for the CUDA-graph R-bucketing path. Kept free
// of CUDA/Triton/torch so they can be unit-tested on any host (see
// test/cuda_graph_bucketing_test.cc). The backend calls these, so the tested
// logic is the shipped logic.

// Parse a comma-separated int64 list (e.g. the CUDA_GRAPH_BATCH_SIZES
// parameter) into a sorted, deduplicated set. Non-numeric or empty tokens are
// skipped (tolerant parse) and a trailing comma is ignored. Behaviourally
// identical to the historical inline parse in ModelState::ParseParameters.
inline std::set<int64_t>
ParseCsvInt64Set(const std::string& csv)
{
  std::set<int64_t> out;
  size_t start = 0;
  while (start < csv.size()) {
    size_t comma = csv.find(',', start);
    std::string tok = csv.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    try {
      out.insert(std::stoll(tok));
    }
    catch (...) {
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

// Number of padded (dummy) request rows added when a batch of size `r` replays
// the graph captured for R=`bucket` (bucket >= r). Zero when the batch already
// equals the bucket or exceeds it. Used for the cudagraph_pad_waste_rows_total
// metric.
inline int64_t
PadWasteRows(int64_t r, int64_t bucket)
{
  return bucket > r ? bucket - r : 0;
}

}  // namespace triton::backend::pytorch
