/*
 * Copyright (c) 2021 Cyrille Rossant and contributors. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 * SPDX-License-Identifier: MIT
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <volk.h>

#include "_alloc.h"
#include "_assertions.h"
#include "_images.h"
#include "_device.h"
#include "_memory.h"
#include "_log.h"
#include "datoviz/vklite/images.h"
#include "datoviz/vklite/sparse.h"
#include "datoviz/vk/device.h"
#include "datoviz/vk/queues.h"
#include "vk_mem_alloc.h"

struct DvzSparsePage
{
    VmaAllocation allocation;
    VmaAllocationInfo allocation_info;
    uint64_t last_used;
    bool resident;
};

typedef struct DvzSparsePage DvzSparsePage;

struct DvzSparseImage
{
    DvzDevice* device;
    DvzVma* allocator;
    DvzImages images;
    VkSparseImageMemoryRequirements sparse_req;
    VkDeviceSize page_size;
    uint32_t granularity[3];
    uint32_t page_counts[3];
    uint64_t page_count;
    DvzSparsePage* pages;
    uint64_t memory_budget;
    uint64_t resident_bytes;
    uint64_t clock;
    VkQueue queue;
    VkFence fence;
};

static uint32_t _format_bytes_per_texel(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
        return 1;
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
        return 2;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
        return 4;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
        return 2;
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
        return 4;
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:
        return 8;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_SRGB:
        return 4;
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    default:
        return 0;
    }
}



static uint64_t _page_index(const DvzSparseImage* s, uint32_t x, uint32_t y, uint32_t z)
{
    return ((uint64_t)z * s->page_counts[1] + y) * s->page_counts[0] + x;
}

static bool _region_pages(
    const DvzSparseImage* s, uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t h,
    uint32_t d, uint32_t minp[3], uint32_t maxp[3])
{
    if (w == 0 || h == 0 || d == 0)
        return false;
    uint64_t ex = (uint64_t)x + w, ey = (uint64_t)y + h, ez = (uint64_t)z + d;
    uint32_t extent[3] = {s->images.info.extent.width, s->images.info.extent.height,
                          s->images.info.extent.depth};
    if (ex > extent[0] || ey > extent[1] || ez > extent[2])
        return false;
    minp[0] = x / s->granularity[0];
    minp[1] = y / s->granularity[1];
    minp[2] = z / s->granularity[2];
    maxp[0] = (uint32_t)((ex - 1) / s->granularity[0]);
    maxp[1] = (uint32_t)((ey - 1) / s->granularity[1]);
    maxp[2] = (uint32_t)((ez - 1) / s->granularity[2]);
    return true;
}

static bool _allocate_page(DvzSparseImage* s, DvzSparsePage* page)
{
    VkMemoryRequirements req = {0};
    vkGetImageMemoryRequirements(s->device->vk_device, dvz_sparse_image_handle(s), &req);
    req.size = s->page_size;
    req.alignment = s->page_size;
    VmaAllocationCreateInfo ci = {0};
    ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ci.memoryTypeBits = req.memoryTypeBits;
    ci.flags = VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;
    VkResult res = vmaAllocateMemory(s->allocator->vma, &req, &ci, &page->allocation,
                                     &page->allocation_info);
    if (res != VK_SUCCESS)
    {
        log_error("sparse page allocation failed (VkResult=%d)", (int)res);
        return false;
    }
    page->resident = true;
    page->last_used = ++s->clock;
    s->resident_bytes += s->page_size;
    return true;
}

static bool _bind_pages(DvzSparseImage* s, VkSparseImageMemoryBind* binds, uint32_t count)
{
    if (count == 0)
        return true;
    VkSparseImageMemoryBindInfo image_bind = {
        .image = dvz_sparse_image_handle(s), .bindCount = count, .pBinds = binds};
    VkBindSparseInfo info = {
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .imageBindCount = 1, .pImageBinds = &image_bind};
    vkResetFences(s->device->vk_device, 1, &s->fence);
    VkResult res = vkQueueBindSparse(s->queue, 1, &info, s->fence);
    if (res != VK_SUCCESS)
    {
        log_error("vkQueueBindSparse failed (VkResult=%d)", (int)res);
        return false;
    }
    res = vkWaitForFences(s->device->vk_device, 1, &s->fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS)
    {
        log_error("waiting for sparse binding failed (VkResult=%d)", (int)res);
        vkDeviceWaitIdle(s->device->vk_device);
        return false;
    }
    return true;
}

static bool _evict_one(DvzSparseImage* s, uint64_t protected_index)
{
    uint64_t best = UINT64_MAX;
    uint64_t best_age = UINT64_MAX;
    for (uint64_t i = 0; i < s->page_count; i++)
    {
        DvzSparsePage* p = &s->pages[i];
        if (!p->resident || i == protected_index)
            continue;
        if (p->last_used < best_age)
        {
            best_age = p->last_used;
            best = i;
        }
    }
    if (best == UINT64_MAX)
        return false;
    uint32_t px = (uint32_t)(best % s->page_counts[0]);
    uint64_t q = best / s->page_counts[0];
    uint32_t py = (uint32_t)(q % s->page_counts[1]);
    uint32_t pz = (uint32_t)(q / s->page_counts[1]);
    VkSparseImageMemoryBind bind = {0};
    bind.subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bind.subresource.mipLevel = 0;
    bind.offset.x = (int32_t)(px * s->granularity[0]);
    bind.offset.y = (int32_t)(py * s->granularity[1]);
    bind.offset.z = (int32_t)(pz * s->granularity[2]);
    bind.extent.width = s->granularity[0];
    bind.extent.height = s->granularity[1];
    bind.extent.depth = s->granularity[2];
    bind.memory = VK_NULL_HANDLE;
    if (!_bind_pages(s, &bind, 1))
        return false;
    vmaFreeMemory(s->allocator->vma, s->pages[best].allocation);
    s->pages[best] = (DvzSparsePage){0};
    s->resident_bytes -= s->page_size;
    return true;
}

DvzSparseImage* dvz_sparse_image_create(
    DvzDevice* device, DvzVma* allocator, VkFormat format, uint32_t width, uint32_t height,
    uint32_t depth, VkImageUsageFlags usage, uint64_t memory_budget)
{
    ANN(device); ANN(allocator);
    const VkPhysicalDeviceFeatures* features = dvz_device_features10(device);
    if (features == NULL || !features->sparseBinding || !features->sparseResidencyImage3D)
    {
        log_error("sparse 3D image requested on a device without sparse residency support");
        return NULL;
    }
    DvzQueue* sparse_queue = dvz_device_queue(device, DVZ_QUEUE_SPARSE);
    if (sparse_queue == NULL)
        sparse_queue = dvz_device_queue(device, DVZ_QUEUE_MAIN);
    if (sparse_queue == NULL || (sparse_queue->flags & VK_QUEUE_SPARSE_BINDING_BIT) == 0)
    {
        log_error("sparse 3D image requested but no sparse-binding queue is available");
        return NULL;
    }
    DvzSparseImage* s = dvz_calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->device = device;
    s->allocator = allocator;
    s->memory_budget = memory_budget;

    dvz_images(device, allocator, VK_IMAGE_TYPE_3D, 1, &s->images);
    dvz_images_format(&s->images, format);
    dvz_images_size(&s->images, width, height, depth);
    dvz_images_usage(&s->images, usage);
    dvz_images_flags(&s->images, VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT);
    if (dvz_images_create(&s->images) != 0)
        goto fail;

    uint32_t n = 0;
    vkGetImageSparseMemoryRequirements(device->vk_device, dvz_image_handle(&s->images, 0), &n, NULL);
    if (n == 0) goto fail_image;
    VkSparseImageMemoryRequirements* reqs = dvz_calloc(n, sizeof(*reqs));
    if (!reqs) goto fail_image;
    vkGetImageSparseMemoryRequirements(device->vk_device, dvz_image_handle(&s->images, 0), &n, reqs);
    if (n != 1 || reqs[0].formatProperties.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
        reqs[0].formatProperties.flags != 0 || reqs[0].imageMipTailSize != 0)
    {
        log_error("unsupported sparse 3D image layout: only standard single-mip color images are supported");
        dvz_free(reqs);
        goto fail_image;
    }
    s->sparse_req = reqs[0];
    dvz_free(reqs);
    s->granularity[0] = s->sparse_req.formatProperties.imageGranularity.width;
    s->granularity[1] = s->sparse_req.formatProperties.imageGranularity.height;
    s->granularity[2] = s->sparse_req.formatProperties.imageGranularity.depth;
    if (!s->granularity[0] || !s->granularity[1] || !s->granularity[2]) goto fail_image;

    VkMemoryRequirements image_memory_req = {0};
    vkGetImageMemoryRequirements(device->vk_device, dvz_image_handle(&s->images, 0), &image_memory_req);
    uint32_t texel_bytes = _format_bytes_per_texel(format);
    uint64_t page_bytes = (uint64_t)s->granularity[0] * s->granularity[1] * s->granularity[2] * texel_bytes;
    if (texel_bytes == 0 || page_bytes == 0 || image_memory_req.alignment == 0) goto fail_image;
    uint64_t aligned_page_bytes = (page_bytes + image_memory_req.alignment - 1) / image_memory_req.alignment * image_memory_req.alignment;
    if (aligned_page_bytes > UINT64_MAX || aligned_page_bytes > (uint64_t)SIZE_MAX) goto fail_image;
    s->page_size = (VkDeviceSize)aligned_page_bytes;
    s->page_counts[0] = (width + s->granularity[0] - 1) / s->granularity[0];
    s->page_counts[1] = (height + s->granularity[1] - 1) / s->granularity[1];
    s->page_counts[2] = (depth + s->granularity[2] - 1) / s->granularity[2];
    s->page_count = (uint64_t)s->page_counts[0] * s->page_counts[1] * s->page_counts[2];
    if (s->page_count == 0 || s->page_count > SIZE_MAX / sizeof(DvzSparsePage)) goto fail_image;
    s->pages = dvz_calloc(s->page_count, sizeof(*s->pages));
    if (!s->pages) goto fail_image;
    s->queue = sparse_queue != NULL ? dvz_queue_handle(sparse_queue) : VK_NULL_HANDLE;
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (s->queue == VK_NULL_HANDLE || vkCreateFence(device->vk_device, &fci, NULL, &s->fence) != VK_SUCCESS)
        goto fail_pages;
    return s;

fail_pages:
    dvz_free(s->pages);
fail_image:
    dvz_images_destroy(&s->images);
fail:
    dvz_free(s);
    return NULL;
}

void dvz_sparse_image_destroy(DvzSparseImage* s)
{
    if (!s) return;
    if (s->fence != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(s->device->vk_device);
        vkDestroyFence(s->device->vk_device, s->fence, NULL);
    }
    if (s->pages)
    {
        for (uint64_t i = 0; i < s->page_count; i++)
            if (s->pages[i].resident) vmaFreeMemory(s->allocator->vma, s->pages[i].allocation);
        dvz_free(s->pages);
    }
    dvz_images_destroy(&s->images);
    dvz_free(s);
}

VkImage dvz_sparse_image_handle(DvzSparseImage* s) { ANN(s); return dvz_image_handle(&s->images, 0); }
void dvz_sparse_image_granularity(DvzSparseImage* s, uint32_t out[3]) { ANN(s); ANN(out); memcpy(out, s->granularity, sizeof(s->granularity)); }
VkDeviceSize dvz_sparse_image_page_size(DvzSparseImage* s) { ANN(s); return s->page_size; }
uint64_t dvz_sparse_image_resident_pages(DvzSparseImage* s) { ANN(s); return s->resident_bytes / s->page_size; }

bool dvz_sparse_image_ensure_region(
    DvzSparseImage* s, uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t h, uint32_t d)
{
    ANN(s);
    uint32_t minp[3], maxp[3];
    if (!_region_pages(s, x, y, z, w, h, d, minp, maxp)) return false;
    uint64_t needed = 0;
    for (uint32_t pz=minp[2]; pz<=maxp[2]; pz++) for (uint32_t py=minp[1]; py<=maxp[1]; py++) for (uint32_t px=minp[0]; px<=maxp[0]; px++)
        if (!s->pages[_page_index(s,px,py,pz)].resident) needed++;
    if (s->memory_budget != 0 && needed * s->page_size > s->memory_budget) return false;
    while (s->memory_budget != 0 && s->resident_bytes + needed * s->page_size > s->memory_budget)
        if (!_evict_one(s, UINT64_MAX)) return false;

    uint64_t max_binds = needed;
    if (max_binds > UINT32_MAX) return false;
    VkSparseImageMemoryBind* binds = needed ? dvz_calloc(max_binds, sizeof(*binds)) : NULL;
    if (needed && !binds) return false;
    uint32_t bind_count = 0;
    for (uint32_t pz=minp[2]; pz<=maxp[2]; pz++) for (uint32_t py=minp[1]; py<=maxp[1]; py++) for (uint32_t px=minp[0]; px<=maxp[0]; px++)
    {
        uint64_t idx = _page_index(s,px,py,pz); DvzSparsePage* p=&s->pages[idx];
        if (p->resident) { p->last_used=++s->clock; continue; }
        if (!_allocate_page(s,p)) { dvz_free(binds); return false; }
        VkSparseImageMemoryBind* b=&binds[bind_count++];
        b->subresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; b->subresource.mipLevel=0;
        b->offset.x=(int32_t)(px*s->granularity[0]); b->offset.y=(int32_t)(py*s->granularity[1]); b->offset.z=(int32_t)(pz*s->granularity[2]);
        b->extent.width=s->granularity[0]; b->extent.height=s->granularity[1]; b->extent.depth=s->granularity[2];
        b->memory=p->allocation_info.deviceMemory; b->memoryOffset=p->allocation_info.offset;
    }
    bool ok = _bind_pages(s, binds, bind_count);
    if (!ok)
    {
        /* The bind did not reach the device. Release only the allocations introduced by this
         * operation; pages that were already resident were never added to `binds`. */
        for (uint32_t i = 0; i < bind_count; i++)
        {
            const VkSparseImageMemoryBind* b = &binds[i];
            uint32_t px = (uint32_t)b->offset.x / s->granularity[0];
            uint32_t py = (uint32_t)b->offset.y / s->granularity[1];
            uint32_t pz = (uint32_t)b->offset.z / s->granularity[2];
            uint64_t idx = _page_index(s, px, py, pz);
            if (idx < s->page_count && s->pages[idx].resident)
            {
                vmaFreeMemory(s->allocator->vma, s->pages[idx].allocation);
                s->pages[idx] = (DvzSparsePage){0};
                s->resident_bytes -= s->page_size;
            }
        }
    }
    dvz_free(binds);
    return ok;
}

void dvz_sparse_image_touch_region(DvzSparseImage* s, uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t h, uint32_t d)
{
    ANN(s); uint32_t minp[3],maxp[3]; if (!_region_pages(s,x,y,z,w,h,d,minp,maxp)) return;
    for(uint32_t pz=minp[2];pz<=maxp[2];pz++) for(uint32_t py=minp[1];py<=maxp[1];py++) for(uint32_t px=minp[0];px<=maxp[0];px++) { DvzSparsePage*p=&s->pages[_page_index(s,px,py,pz)]; if(p->resident)p->last_used=++s->clock; }
}

bool dvz_sparse_image_page_resident(DvzSparseImage* s,uint32_t x,uint32_t y,uint32_t z)
{
    ANN(s); if(x>=s->page_counts[0]||y>=s->page_counts[1]||z>=s->page_counts[2]) return false; return s->pages[_page_index(s,x,y,z)].resident;
}
