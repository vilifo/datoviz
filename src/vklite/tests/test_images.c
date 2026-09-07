/*
 * Copyright (c) 2021 Cyrille Rossant and contributors. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 * SPDX-License-Identifier: MIT
 */

/*************************************************************************************************/
/*  Testing images                                                                               */
/*************************************************************************************************/



/*************************************************************************************************/
/*  Includes                                                                                     */
/*************************************************************************************************/

#include "test_vk.h"
#include "_assertions.h"
#include "datoviz/vk/device.h"
#include "datoviz/vk/gpu_ctx.h"
#include "datoviz/vklite/images.h"
#include "datoviz/vklite/sparse.h"
#include "test_vklite.h"
#include "testing.h"
#include "vulkan_core.h"

#include <volk.h>



/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

#define MAP_OFFSET 64
#define MAP_SIZE   1024



/*************************************************************************************************/
/*  Tests                                                                                        */
/*************************************************************************************************/

int test_vklite_images_1(TstContext* suite, const TstCase* tstitem)
{
    ANN(suite);
    ANN(tstitem);

    // Bootstrap.
    DvzGpuCtxConfig cfg = dvz_testing_gpu_ctx_config(suite);
    DvzGpuCtx* ctx = dvz_gpu_ctx(&cfg);
    ANN(ctx);

    // Images.
    DvzImages* images = dvz_images_create_wrapper();
    ANN(images);
    dvz_images(
        dvz_gpu_ctx_device(ctx), dvz_gpu_ctx_alloc(ctx), VK_IMAGE_TYPE_2D, 1, images);
    dvz_images_format(images, VK_FORMAT_R8G8B8A8_UNORM);
    dvz_images_size(images, 256, 256, 1);
    dvz_images_mip(images, 1);
    dvz_images_layers(images, 2);
    dvz_images_samples(images, VK_SAMPLE_COUNT_1_BIT);
    dvz_images_usage(images, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    AT(dvz_images_count(images) == 1);
    AT(dvz_images_format_value(images) == VK_FORMAT_R8G8B8A8_UNORM);
    dvz_images_create(images);
    AT(dvz_image_handle(images, 0) != VK_NULL_HANDLE);

    // Image views.
    DvzImageViews* views = dvz_image_views_create_wrapper();
    ANN(views);
    dvz_image_views(images, views);
    dvz_image_views_create(views);
    AT(dvz_image_views_count(views) == 1);
    AT(dvz_image_views_handle(views, 0) != VK_NULL_HANDLE);

    // Cleanup.
    dvz_image_views_destroy(views);
    dvz_image_views_destroy(views);
    dvz_images_destroy(images);
    dvz_images_destroy(images);
    AT(dvz_image_views_handle(views, 0) == VK_NULL_HANDLE);
    AT(dvz_image_handle(images, 0) == VK_NULL_HANDLE);

    AT(dvz_images_create(images) == 0);
    AT(dvz_image_handle(images, 0) != VK_NULL_HANDLE);
    dvz_image_views_create(views);
    AT(dvz_image_views_handle(views, 0) != VK_NULL_HANDLE);
    dvz_image_views_destroy(views);
    dvz_images_destroy(images);
    AT(dvz_image_views_handle(views, 0) == VK_NULL_HANDLE);
    AT(dvz_image_handle(images, 0) == VK_NULL_HANDLE);

    DvzImageCopy* copy = dvz_image_copy_create();
    DvzImageBlit* blit = dvz_image_blit_create();
    ANN(copy);
    ANN(blit);
    dvz_image_copy(copy);
    dvz_image_blit(blit);
    dvz_image_copy_free(copy);
    dvz_image_blit_free(blit);

    dvz_image_views_free(views);
    dvz_images_free(images);
    uint32_t err_count = dvz_gpu_ctx_error_count(ctx);
    dvz_gpu_ctx_destroy(ctx);

    return err_count > 0;
}



int test_vklite_sparse_image_1(TstContext* suite, const TstCase* tstitem)
{
    ANN(suite);
    ANN(tstitem);

    // Bootstrap.
    DvzGpuCtxConfig cfg = dvz_testing_gpu_ctx_config(suite);
    DvzGpuCtx* ctx = dvz_gpu_ctx(&cfg);
    ANN(ctx);

    DvzDevice* device = dvz_gpu_ctx_device(ctx);
    VkPhysicalDevice pdevice = VK_NULL_HANDLE;
    DvzInstance* instance = dvz_gpu_ctx_instance(ctx);
    uint32_t gpu_index = dvz_gpu_ctx_gpu_index(ctx);
    bool sparse_supported =
        instance != NULL && gpu_index != UINT32_MAX &&
        dvz_instance_gpu_handle(instance, gpu_index, &pdevice);
    VkPhysicalDeviceFeatures features = {0};
    if (sparse_supported)
    {
        vkGetPhysicalDeviceFeatures(pdevice, &features);
        sparse_supported = features.sparseBinding && features.sparseResidencyImage3D;
    }
    if (!sparse_supported)
    {
        tst_skip(suite, "sparse 3D image residency is unavailable on the selected device");
        dvz_gpu_ctx_destroy(ctx);
        return 0;
    }

    // Make the logical image exactly two pages wide/high/deep. One page of memory is enough
    // to force the residency manager to evict the previous page when the second page is needed.
    DvzSparseImage* probe = dvz_sparse_image_create(
        device, dvz_gpu_ctx_alloc(ctx), VK_FORMAT_R8_UNORM, 64, 64, 64,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
    ANN(probe);
    uint32_t g[3] = {0};
    dvz_sparse_image_granularity(probe, g);
    AT(g[0] > 0 && g[1] > 0 && g[2] > 0);
    VkDeviceSize page_size = dvz_sparse_image_page_size(probe);
    AT(page_size > 0);
    dvz_sparse_image_destroy(probe);

    uint32_t width = 2 * g[0];
    uint32_t height = 2 * g[1];
    uint32_t depth = 2 * g[2];
    DvzSparseImage* sparse = dvz_sparse_image_create(
        device, dvz_gpu_ctx_alloc(ctx), VK_FORMAT_R8_UNORM, width, height, depth,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, page_size);
    ANN(sparse);

    // A new sparse image starts completely non-resident.
    AT(dvz_sparse_image_resident_pages(sparse) == 0);
    AT(!dvz_sparse_image_page_resident(sparse, 0, 0, 0));

    // Bind the first page through vkQueueBindSparse.
    AT(dvz_sparse_image_ensure_region(sparse, 0, 0, 0, 1, 1, 1));
    AT(dvz_sparse_image_resident_pages(sparse) == 1);
    AT(dvz_sparse_image_page_resident(sparse, 0, 0, 0));

    // Re-requesting the same page must not allocate/bind a second page.
    AT(dvz_sparse_image_ensure_region(sparse, 1, 1, 1, 1, 1, 1));
    AT(dvz_sparse_image_resident_pages(sparse) == 1);
    AT(dvz_sparse_image_page_resident(sparse, 0, 0, 0));

    // Request a different page. The one-page budget must evict the old page and bind the new one.
    AT(dvz_sparse_image_ensure_region(sparse, g[0], 0, 0, 1, 1, 1));
    AT(dvz_sparse_image_resident_pages(sparse) == 1);
    AT(!dvz_sparse_image_page_resident(sparse, 0, 0, 0));
    AT(dvz_sparse_image_page_resident(sparse, 1, 0, 0));

    // Request the original page again; this exercises another unbind + bind cycle.
    AT(dvz_sparse_image_ensure_region(sparse, 0, 0, 0, 1, 1, 1));
    AT(dvz_sparse_image_resident_pages(sparse) == 1);
    AT(dvz_sparse_image_page_resident(sparse, 0, 0, 0));
    AT(!dvz_sparse_image_page_resident(sparse, 1, 0, 0));

    dvz_sparse_image_destroy(sparse);
    uint32_t err_count = dvz_gpu_ctx_error_count(ctx);
    dvz_gpu_ctx_destroy(ctx);
    return err_count > 0;
}

int test_vklite_images_create_requires_destroy(TstContext* suite, const TstCase* tstitem)
{
    ANN(suite);
    ANN(tstitem);

    DvzGpuCtxConfig cfg = dvz_testing_gpu_ctx_config(suite);
    DvzGpuCtx* ctx = dvz_gpu_ctx(&cfg);
    ANN(ctx);

    DvzImages* images = dvz_images_create_wrapper();
    ANN(images);
    dvz_images(
        dvz_gpu_ctx_device(ctx), dvz_gpu_ctx_alloc(ctx), VK_IMAGE_TYPE_2D, 1, images);
    dvz_images_format(images, VK_FORMAT_R8G8B8A8_UNORM);
    dvz_images_mip(images, 1);
    dvz_images_layers(images, 1);
    dvz_images_samples(images, VK_SAMPLE_COUNT_1_BIT);
    dvz_images_usage(images, VK_IMAGE_USAGE_SAMPLED_BIT);

    VkPhysicalDeviceProperties props = {0};
    vkGetPhysicalDeviceProperties(
        dvz_device_physical_device(dvz_gpu_ctx_device(ctx)), &props);
    dvz_images_size(images, props.limits.maxImageDimension2D + 1, 16, 1);

    AT_EXPECTED_ERROR_STRICT(suite, dvz_images_create(images) != 0);
    AT(dvz_image_handle(images, 0) == VK_NULL_HANDLE);

    DvzImageViews* views = dvz_image_views_create_wrapper();
    ANN(views);
    dvz_image_views(images, views);
    tst_expect_error_begin(suite);
    dvz_image_views_create(views);
    AT(tst_expect_error_end(suite) == 0);
    AT(dvz_image_views_handle(views, 0) == VK_NULL_HANDLE);

    dvz_image_views_destroy(views);
    dvz_images_destroy(images);
    AT(dvz_image_handle(images, 0) == VK_NULL_HANDLE);

    dvz_images_size(images, 64, 64, 1);
    AT(dvz_images_create(images) == 0);
    AT(dvz_image_handle(images, 0) != VK_NULL_HANDLE);

    AT_EXPECTED_ERROR_STRICT(suite, dvz_images_create(images) != 0);
    AT(dvz_image_handle(images, 0) != VK_NULL_HANDLE);

    dvz_image_views_create(views);
    AT(dvz_image_views_handle(views, 0) != VK_NULL_HANDLE);
    tst_expect_error_begin(suite);
    dvz_image_views_create(views);
    AT(tst_expect_error_end(suite) == 0);
    AT(dvz_image_views_handle(views, 0) != VK_NULL_HANDLE);

    dvz_image_views_destroy(views);
    dvz_images_destroy(images);
    dvz_image_views_free(views);
    dvz_images_free(images);
    uint32_t err_count = dvz_gpu_ctx_error_count(ctx);
    dvz_gpu_ctx_destroy(ctx);

    return err_count > 0;
}
