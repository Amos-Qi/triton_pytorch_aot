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

#include "cuda_graph_bucketing.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>

namespace triton::backend::pytorch {
namespace {

TEST(ParseCsvInt64Set, ParsesSortedUnique)
{
  EXPECT_EQ(ParseCsvInt64Set("16,18,20,22,24"), (std::set<int64_t>{16, 18, 20, 22, 24}));
}

TEST(ParseCsvInt64Set, SortsAndDeduplicates)
{
  EXPECT_EQ(ParseCsvInt64Set("24,16,16,20"), (std::set<int64_t>{16, 20, 24}));
}

TEST(ParseCsvInt64Set, SingleValue)
{
  EXPECT_EQ(ParseCsvInt64Set("16"), (std::set<int64_t>{16}));
}

TEST(ParseCsvInt64Set, EmptyStringYieldsEmptySet)
{
  EXPECT_TRUE(ParseCsvInt64Set("").empty());
}

TEST(ParseCsvInt64Set, ToleratesSurroundingWhitespace)
{
  // std::stoll skips leading whitespace and stops at the first non-digit.
  EXPECT_EQ(ParseCsvInt64Set(" 16 , 20 "), (std::set<int64_t>{16, 20}));
}

TEST(ParseCsvInt64Set, SkipsEmptyAndTrailingCommaTokens)
{
  EXPECT_EQ(ParseCsvInt64Set("16,,20,"), (std::set<int64_t>{16, 20}));
  EXPECT_EQ(ParseCsvInt64Set(",16"), (std::set<int64_t>{16}));
}

TEST(ParseCsvInt64Set, SkipsNonNumericTokens)
{
  EXPECT_EQ(ParseCsvInt64Set("16,abc,20"), (std::set<int64_t>{16, 20}));
}

TEST(PadWasteRows, ZeroWhenExactOrRequestExceedsBucket)
{
  EXPECT_EQ(PadWasteRows(16, 16), 0);
  EXPECT_EQ(PadWasteRows(20, 16), 0);  // r > bucket must never yield negative waste
}

TEST(PadWasteRows, DifferenceWhenPaddedUp)
{
  EXPECT_EQ(PadWasteRows(9, 16), 7);
  EXPECT_EQ(PadWasteRows(1, 16), 15);
  EXPECT_EQ(PadWasteRows(15, 16), 1);
}

}  // namespace
}  // namespace triton::backend::pytorch
