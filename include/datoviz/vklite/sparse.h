/*
 * Copyright (c) 2021 Cyrille Rossant and contributors. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "datoviz/common/macros.h"
#include "datoviz/vk/memory.h"

EXTERN_C_ON

typedef struct DvzDevice DvzDevice;
typedef struct DvzSparseImage DvzSparseImage;

/**
 * Create a sparse-resident 3D image and its CPU-side page residency manager.
 *
 * Only single-mip, standard sparse-image formats are currently supported. Memory is allocated
 * per sparse page and bound with vkQueueBindSparse; the normal vkBindImageMemory path is never
 * used for the image.
 */
DVZ_EXPORT DvzSparseImage* dvz_sparse_image_create(
    DvzDevice* device, DvzVma* allocator, VkFormat format, uint32_t width, uint32_t height,
    uint32_t depth, VkImageUsageFlags usage, uint64_t memory_budget);

/** Destroy a sparse image and all resident page allocations. */
DVZ_EXPORT void dvz_sparse_image_destroy(DvzSparseImage* sparse);

/** Return the logical Vulkan image handle. */
DVZ_EXPORT VkImage dvz_sparse_image_handle(DvzSparseImage* sparse);

/** Return the page granularity in texels. */
DVZ_EXPORT void dvz_sparse_image_granularity(DvzSparseImage* sparse, uint32_t out_granularity[3]);

/** Return the sparse page size in bytes. */
DVZ_EXPORT VkDeviceSize dvz_sparse_image_page_size(DvzSparseImage* sparse);

/** Return the current number of resident pages. */
DVZ_EXPORT uint64_t dvz_sparse_image_resident_pages(DvzSparseImage* sparse);

/**
 * Make every sparse page intersecting a logical region resident.
 *
 * The operation is batched into one vkQueueBindSparse call and waits for that binding to finish
 * before returning. Pages outside the requested region may be evicted according to LRU when the
 * configured memory budget is exceeded.
 */
DVZ_EXPORT bool dvz_sparse_image_ensure_region(
    DvzSparseImage* sparse, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
    uint32_t depth);

/** Mark all pages intersecting a logical region as recently used. */
DVZ_EXPORT void dvz_sparse_image_touch_region(
    DvzSparseImage* sparse, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
    uint32_t depth);

/** Return whether the page at a sparse-grid coordinate is currently resident. */
DVZ_EXPORT bool dvz_sparse_image_page_resident(
    DvzSparseImage* sparse, uint32_t page_x, uint32_t page_y, uint32_t page_z);

EXTERN_C_OFF
