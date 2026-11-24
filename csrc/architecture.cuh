// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NVCC preprocesses each -gencode device target separately. Rubin requires
// different multicast operand widths and MMA instructions from Blackwell.
// Only device helpers depend on this selection: MoK's host-visible descriptors,
// kernel argument types, tile sizes, and shared-memory allocation stay identical.
#ifdef MOK_ARCH_ALL
#if defined(KITTENS_SM100) || defined(KITTENS_SM103) || defined(KITTENS_SM107)
#error "MOK_ARCH_ALL selects the ThunderKittens architecture; do not also set KITTENS_SM*"
#endif
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1000
#define KITTENS_SM100
#elif __CUDA_ARCH__ == 1030
#define KITTENS_SM103
#elif __CUDA_ARCH__ == 1070
#define KITTENS_SM107
#else
#error "MoK supports only SM100, SM103 and SM107 device targets"
#endif
#endif
