/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/conv/invokers/impl_gemm.hpp>
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>

#include "implicitgemm_util.hpp"

#include <cstddef>

namespace miopen {
namespace solver {
PerformanceImplicitGemmMlir::PerformanceImplicitGemmMlir(int BlockSize_,
                                                         int GemmMPerBlock_,
                                                         int GemmNPerBlock_,
                                                         int GemmKPerBlock_,
                                                         int GemmMPerThread_,
                                                         int GemmNPerThread_,
                                                         bool use_spare_set_)
    : BlockSize(BlockSize_),
      GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmMPerThread(GemmMPerThread_),
      GemmNPerThread(GemmNPerThread_),
      use_spare_set(use_spare_set_)
{
}

PerformanceImplicitGemmMlir::PerformanceImplicitGemmMlir(bool spare)
{
    // always search full space, no matter if use_spare_set or not
    BlockSize = 64;

    GemmMPerBlock = 32;
    GemmNPerBlock = 32;
    GemmKPerBlock = 4;

    GemmMPerThread = 2;
    GemmNPerThread = 2;

    use_spare_set = spare;
}

bool PerformanceImplicitGemmMlir::operator==(const PerformanceImplicitGemmMlir& other) const
{
    // clang-format off
    return BlockSize == other.BlockSize
        && GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmMPerThread == other.GemmMPerThread
        && GemmNPerThread == other.GemmNPerThread
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

std::tuple<int, bool>
PerformanceImplicitGemmMlir::CalculateGridSize(const ConvolutionContext& ctx) const
{
    int GridSize = 0;

    try
    {
        int gemm_m = 0;
        int gemm_n = 0;

        std::tie(gemm_m, gemm_n, std::ignore) = ConvHipImplicitGemmMlir::CalculateGemmSize(ctx);

        if(!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0))
            MIOPEN_THROW("invalid performance parameter");

        GridSize = (gemm_m / GemmMPerBlock) * (gemm_n / GemmNPerBlock);
    }
    catch(...)
    {
        return std::make_tuple(-1, false);
    }

    return std::make_tuple(GridSize, true);
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmMlir::CalculateBlockGemmPerformanceParameters(
    const ConvolutionContext&) const
{
    int GemmMLevel0Cluster = 0;
    int GemmNLevel0Cluster = 0;
    int GemmMLevel1Cluster = 0;
    int GemmNLevel1Cluster = 0;

    try
    {
        if(BlockSize == 64)
        {
            GemmMLevel0Cluster = 4;
            GemmNLevel0Cluster = 4;
            GemmMLevel1Cluster = 2;
            GemmNLevel1Cluster = 2;
        }
        else if(BlockSize == 128)
        {
            GemmMLevel0Cluster = 4;
            GemmNLevel0Cluster = 4;
            GemmMLevel1Cluster = 4;
            GemmNLevel1Cluster = 2;
        }
        else if(BlockSize == 256)
        {
            GemmMLevel0Cluster = 4;
            GemmNLevel0Cluster = 4;
            GemmMLevel1Cluster = 4;
            GemmNLevel1Cluster = 4;
        }
        else
        {
            MIOPEN_LOG_E("BlockSize not supported");
            MIOPEN_THROW("invalid performance parameter");
        }

        if(!(GemmMPerBlock % GemmMPerThread == 0 && GemmNPerBlock % GemmNPerThread == 0))
            MIOPEN_THROW("invalid performance parameter");

        const auto thread_gemm_per_block_m = GemmMPerBlock / GemmMPerThread;
        const auto thread_gemm_per_block_n = GemmNPerBlock / GemmNPerThread;

        const auto thread_gemm_per_cluster_m = GemmMLevel0Cluster * GemmMLevel1Cluster;
        const auto thread_gemm_per_cluster_n = GemmNLevel0Cluster * GemmNLevel1Cluster;

        if(!(thread_gemm_per_block_m % thread_gemm_per_cluster_m == 0) &&
           (thread_gemm_per_block_n % thread_gemm_per_cluster_n == 0))
            MIOPEN_THROW("invalid performance parameter");

        const auto cluster_per_block_m = thread_gemm_per_block_m / thread_gemm_per_cluster_m;
        const auto cluster_per_block_n = thread_gemm_per_block_n / thread_gemm_per_cluster_n;

        // inline asm only support cluster_per_block_m = 2 andcluster_per_block_n = 2
        if(!(cluster_per_block_m == 2 && cluster_per_block_n == 2))
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        return std::make_tuple(-1, -1, -1, -1, false);
    }

    return std::make_tuple(
        GemmMLevel0Cluster, GemmNLevel0Cluster, GemmMLevel1Cluster, GemmNLevel1Cluster, true);
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmMlir::CalculateGemmABlockCopyPerformanceParameters(
    const ConvolutionContext&) const
{
    int ClusterLengths_GemmK  = 0;
    int ClusterLengths_GemmM  = 0;
    int SrcDataPerRead_GemmK  = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmM = amd_lds_write_max_length<float>();

    try
    {
        // calculate vector length on gemmk dimension
        SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, GemmKPerBlock);

        // calculate threadwise copy size
        const auto a_data_per_thread_copy = (GemmKPerBlock * GemmMPerBlock) / BlockSize;

        if(!(a_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmABlockCopySrcDataPerRead_GemmK also bounded by size of threadwise copy
        SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, a_data_per_thread_copy);

        // decide threadwise copy lengths
        const auto a_data_per_thread_copy_gemmk = SrcDataPerRead_GemmK;
        const auto a_data_per_thread_copy_gemmm =
            a_data_per_thread_copy / a_data_per_thread_copy_gemmk;

        // GemmABlockCopyDstDataPerWrite_GemmM also bounded by size of threadwise copy
        DstDataPerWrite_GemmM = gcd(DstDataPerWrite_GemmM, a_data_per_thread_copy_gemmm);

        // calculate blockwise copy thread cluster lengths
        ClusterLengths_GemmK = GemmKPerBlock / a_data_per_thread_copy_gemmk;
        ClusterLengths_GemmM = GemmMPerBlock / a_data_per_thread_copy_gemmm;

        if(!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmM > 0))
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        return std::make_tuple(-1, -1, -1, -1, false);
    }

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmM,
                           SrcDataPerRead_GemmK,
                           DstDataPerWrite_GemmM,
                           true);
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmMlir::CalculateGemmBBlockCopyPerformanceParameters(
    const ConvolutionContext& ctx) const
{
    int ClusterLengths_GemmK  = 0;
    int ClusterLengths_GemmN  = 0;
    int SrcDataPerRead_GemmN  = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmN = amd_lds_write_max_length<float>();

    try
    {
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

        // calculate vector length on gemmn dimension
        const auto y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
        const auto x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
        const auto hi = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
        const auto wi = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
        const auto conv_stride_h =
            ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
        const auto conv_stride_w =
            ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
        const auto conv_dilation_w =
            ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
        const auto in_left_pad_h  = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
        const auto in_left_pad_w  = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
        const auto in_right_pad_h = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
        const auto in_right_pad_w = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

        if(ctx.Is3d())
        {
            const auto di = ConvolutionContextInterpreter::GetInputDepthDi(ctx);
            const auto z  = ConvolutionContextInterpreter::GetFilterDepthZ(ctx);
            const auto conv_stride_d =
                ConvolutionContextInterpreter::GetAdjustedConvolutionStrideD(ctx);
            const auto in_left_pad_d = ConvolutionContextInterpreter::GetInputLeftPadD(ctx);
            const auto in_right_pad_d =
                ConvolutionContextInterpreter::GetAdjustedInputRightPadD(ctx);

            if(z == 1 && y == 1 && x == 1 && conv_stride_d == 1 && conv_stride_h == 1 &&
               conv_stride_w == 1 && in_left_pad_d == 0 && in_left_pad_h == 0 &&
               in_left_pad_w == 0 && in_right_pad_d == 0 && in_right_pad_h == 0 &&
               in_right_pad_w == 0)
            {
                // \todo there are more configs that can go through this if branch
                SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, di * hi * wi);
            }
            else if(conv_stride_w == 1)
            {
                SrcDataPerRead_GemmN =
                    gcd(SrcDataPerRead_GemmN, in_left_pad_w, wi, in_right_pad_w, conv_dilation_w);
            }
            else
            {
                SrcDataPerRead_GemmN = 1;
            }
        }
        else
        {
            if(y == 1 && x == 1 && conv_stride_h == 1 && conv_stride_w == 1 && in_left_pad_h == 0 &&
               in_left_pad_w == 0 && in_right_pad_h == 0 && in_right_pad_w == 0)
            {
                // \todo there are more configs that can go through this if branch
                SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, hi * wi);
            }
            else if(conv_stride_w == 1)
            {
                SrcDataPerRead_GemmN =
                    gcd(SrcDataPerRead_GemmN, in_left_pad_w, wi, in_right_pad_w, conv_dilation_w);
            }
            else
            {
                SrcDataPerRead_GemmN = 1;
            }
        }

        // calculate threadwise copy size
        const auto b_data_per_thread_copy = (GemmKPerBlock * GemmNPerBlock) / BlockSize;

        if(!(b_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmBBlockCopySrcDataPerRead_GemmN also bounded by size of threadwise copy
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, b_data_per_thread_copy);

        const auto b_data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
        const auto b_data_per_thread_copy_gemmk =
            b_data_per_thread_copy / b_data_per_thread_copy_gemmn;

        // GemmBBlockCopyDstDataPerWrite_GemmN also bounded by size of threadwise copy
        DstDataPerWrite_GemmN = gcd(DstDataPerWrite_GemmN, b_data_per_thread_copy_gemmn);

        // calculate blockwise copy thread cluster lengths
        ClusterLengths_GemmK = GemmKPerBlock / b_data_per_thread_copy_gemmk;
        ClusterLengths_GemmN = GemmNPerBlock / b_data_per_thread_copy_gemmn;

        if(!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmN > 0))
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        MIOPEN_LOG_I("catch");
        return std::make_tuple(-1, -1, -1, -1, false);
    }

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmN,
                           SrcDataPerRead_GemmN,
                           DstDataPerWrite_GemmN,
                           true);
}

std::tuple<int, bool> PerformanceImplicitGemmMlir::CalculateGemmCThreadCopyPerformanceParameters(
    const ConvolutionContext& ctx) const
{
    int DstDataPerWrite_GemmN1 = amd_buffer_store_max_length<float>();

    try
    {
        // GemmCThreadCopyDstDataPerWrite_GemmN1 bounded by size of threadwise GEMM
        DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, GemmNPerThread);

        // GemmCThreadCopyDstDataPerWrite_GemmN1 limited by global memory layout of output tensor
        const auto ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
        const auto wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
        DstDataPerWrite_GemmN1 =
            ctx.Is3d() ? gcd(DstDataPerWrite_GemmN1,
                             ho * wo * ConvolutionContextInterpreter::GetOutputDepthDo(ctx))
                       : gcd(DstDataPerWrite_GemmN1, ho * wo);
    }
    catch(...)
    {
        return std::make_tuple(-1, false);
    }

    return std::make_tuple(DstDataPerWrite_GemmN1, true);
}

std::tuple<std::size_t, bool>
PerformanceImplicitGemmMlir::CalculateLdsNumberOfByte(const ConvolutionContext& ctx) const
{
    std::size_t lds_size = 0;

    try
    {
        bool valid = false;

        int GemmABlockCopyDescDataPerWriteGemmM = 0;
        std::tie(
            std::ignore, std::ignore, std::ignore, GemmABlockCopyDescDataPerWriteGemmM, valid) =
            CalculateGemmABlockCopyPerformanceParameters(ctx);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        int GemmBBlockCopyDescDataPerWriteGemmN = 0;
        std::tie(
            std::ignore, std::ignore, std::ignore, GemmBBlockCopyDescDataPerWriteGemmN, valid) =
            CalculateGemmBBlockCopyPerformanceParameters(ctx);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        const auto ThreadGemmDataPerRead_GemmM = GemmMPerThread;
        const auto ThreadGemmDataPerRead_GemmN = GemmNPerThread;

        const auto max_lds_align = lcm(GemmABlockCopyDescDataPerWriteGemmM,
                                       GemmBBlockCopyDescDataPerWriteGemmN,
                                       ThreadGemmDataPerRead_GemmM,
                                       ThreadGemmDataPerRead_GemmN);

        const auto a_block_space =
            GemmKPerBlock * integer_least_multiple(GemmMPerBlock, max_lds_align);
        const auto b_block_space =
            GemmKPerBlock * integer_least_multiple(GemmNPerBlock, max_lds_align);

        lds_size = 2 * (a_block_space + b_block_space) * sizeof(float);
    }
    catch(...)
    {
        return std::make_tuple(0, false);
    }

    return std::make_tuple(lds_size, true);
}

bool PerformanceImplicitGemmMlir::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<64, 256>(BlockSize) &&
           IsTwoPower<32, 128>(GemmMPerBlock) &&
           IsTwoPower<32, 128>(GemmNPerBlock) &&
           IsTwoPower<4, 16>(GemmKPerBlock) &&
           IsTwoPower<2, 4>(GemmMPerThread) &&
           IsTwoPower<2, 4>(GemmNPerThread);
    // clang-format on
}

bool PerformanceImplicitGemmMlir::IsValid(const ConvolutionContext& ctx) const
{
    if(!IsValidValue())
        return false;

    bool valid = false;

    // check blockwise GEMM size
    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;

    std::tie(gemm_m, gemm_n, gemm_k) = ConvHipImplicitGemmMlir::CalculateGemmSize(ctx);

    if(!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0 && gemm_k % GemmKPerBlock == 0))
        return false;

    if(!(GemmMPerBlock % GemmMPerThread == 0 && GemmNPerBlock % GemmNPerThread == 0))
        return false;

    // check thread cluster in blockwise GEMM
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateBlockGemmPerformanceParameters(ctx);

    if(!valid)
        return false;

    // check blockwise copy of A matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmABlockCopyPerformanceParameters(ctx);

    if(!valid)
        return false;

    // check blockwise copy of B matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmBBlockCopyPerformanceParameters(ctx);

    if(!valid)
        return false;

    // check threadwise copy of C matrix
    std::tie(std::ignore, valid) = CalculateGemmCThreadCopyPerformanceParameters(ctx);

    if(!valid)
        return false;

    // check LDS allocation
    std::size_t lds_size      = 0;
    std::tie(lds_size, valid) = CalculateLdsNumberOfByte(ctx);

    return (valid and lds_size <= get_lds_max_number_of_byte());
}

void PerformanceImplicitGemmMlir::EuristicInit(const ConvolutionContext& ctx)
{
    PerformanceImplicitGemmMlir config;

    config = {256, 128, 128, 16, 4, 4};
    if(!config.IsValid(ctx))
        config = {256, 128, 128, 8, 4, 4};
    if(!config.IsValid(ctx))
        config = {256, 128, 128, 4, 4, 4};
    if(!config.IsValid(ctx))
        config = {128, 128, 64, 16, 4, 4};
    if(!config.IsValid(ctx))
        config = {128, 128, 64, 8, 4, 4};
    if(!config.IsValid(ctx))
        config = {128, 128, 64, 4, 4, 4};
    if(!config.IsValid(ctx))
        config = {128, 64, 128, 16, 4, 4};
    if(!config.IsValid(ctx))
        config = {128, 64, 128, 8, 4, 4};
    if(!config.IsValid(ctx))
        config = {128, 64, 128, 4, 4, 4};
    if(!config.IsValid(ctx))
        config = {64, 64, 64, 16, 4, 4};
    if(!config.IsValid(ctx))
        config = {64, 64, 64, 8, 4, 4};
    if(!config.IsValid(ctx))
        config = {64, 64, 64, 4, 4, 4};
    if(!config.IsValid(ctx))
        config = {64, 64, 32, 16, 4, 2};
    if(!config.IsValid(ctx))
        config = {64, 64, 32, 8, 4, 2};
    if(!config.IsValid(ctx))
        config = {64, 64, 32, 4, 4, 2};
    if(!config.IsValid(ctx))
        config = {64, 32, 64, 16, 2, 4};
    if(!config.IsValid(ctx))
        config = {64, 32, 64, 8, 2, 4};
    if(!config.IsValid(ctx))
        config = {64, 32, 64, 4, 2, 4};
    if(!config.IsValid(ctx))
        config = {64, 32, 32, 16, 2, 2};
    if(!config.IsValid(ctx))
        config = {64, 32, 32, 8, 2, 2};
    if(!config.IsValid(ctx))
        config = {64, 32, 32, 4, 2, 2};
    if(!config.IsValid(ctx))
    {
        MIOPEN_LOG_E("All attempts failed: ");
        assert(false);
    }

    *this = config;
    MIOPEN_LOG_I(ToString());
}

bool PerformanceImplicitGemmMlir::SetNextValue()
{
    // always search full space, no matter if use_spare_set or not
    do
    {
        if(!NextTwoPower<64, 256>(BlockSize))
            break;
        if(!NextTwoPower<32, 128>(GemmMPerBlock))
            break;
        if(!NextTwoPower<32, 128>(GemmNPerBlock))
            break;
        if(!NextTwoPower<4, 16>(GemmKPerBlock))
            break;
        if(!NextTwoPower<2, 4>(GemmMPerThread))
            break;
        if(!NextTwoPower<2, 4>(GemmNPerThread))
            break;

        return false;
    } while(false);

    return true;
}

std::string PerformanceImplicitGemmMlir::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

std::tuple<int, int, int> ConvHipImplicitGemmMlir::CalculateGemmSize(const ConvolutionContext& ctx)
{
    const auto n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const auto k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const auto c  = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const auto ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const auto wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const auto y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const auto x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

    int gemm_m;
    int gemm_n;
    int gemm_k;

    if(ctx.direction.IsForward())
    {
        gemm_m = k;
        gemm_n = n * ho * wo;
        gemm_k = c * y * x;
    }
    else if(ctx.direction.IsBackwardData())
    {
        gemm_m = c * y * x;
        gemm_n = n * ho * wo;
        gemm_k = k;
    }
    else
    {
        gemm_m = k;
        gemm_n = c * y * x;
        gemm_k = n * ho * wo;
    }

    return std::make_tuple(gemm_m, gemm_n, gemm_k);
}

namespace {
bool IsFwApplicable(const ConvolutionContext& ctx)
{
    if(!ctx.direction.IsForward())
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.Is2d() && !ctx.Is3d())
        return false;
    if(!ctx.IsFp32())
        return false;
    if(ctx.group_counts != 1)
        return false;

    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;

    std::tie(gemm_m, gemm_n, gemm_k) = ConvHipImplicitGemmMlir::CalculateGemmSize(ctx);
    return gemm_m % 32 == 0 && gemm_n % 32 == 0 && gemm_k % 4 == 0;
}

bool IsBwdApplicable(const ConvolutionContext& ctx)
{
    if(!ctx.direction.IsBackwardData())
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.Is2d() && !ctx.Is3d())
        return false;
    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;
    if(ctx.group_counts != 1)
        return false;
#if WORKAROUND_ISSUE_309
    if(miopen::HipCompilerVersion() >= external_tool_version_t{3, 5, 0})
        if(ctx.IsBfp16())
            return false;
#endif

    const auto k = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    if(k % GetEPackLength(ctx, false) != 0)
        return false;

    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;

    std::tie(gemm_m, gemm_n, gemm_k) = ConvHipImplicitGemmMlir::CalculateGemmSize(ctx);

    return gemm_m % 32 == 0 && gemm_n % 32 == 0 && gemm_k % 4 == 0;
}

bool IsWrwApplicable(const ConvolutionContext& ctx)
{
    if(ctx.direction.IsForward() || ctx.direction.IsBackwardData())
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.Is2d() && !ctx.Is3d())
        return false;
    if(!ctx.IsFp32())
        return false;
    if(ctx.group_counts != 1)
        return false;

    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;

    std::tie(gemm_m, gemm_n, gemm_k) = ConvHipImplicitGemmMlir::CalculateGemmSize(ctx);
    return gemm_m % 32 == 0 && gemm_n % 32 == 0 && gemm_k % 4 == 0;
}
} // namespace

bool ConvHipImplicitGemmMlir::IsApplicable(const ConvolutionContext& ctx) const
{
    return IsFwApplicable(ctx) || IsBwdApplicable(ctx) || IsWrwApplicable(ctx);
}

PerformanceImplicitGemmMlir
ConvHipImplicitGemmMlir::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmMlir>(ctx);
}

bool ConvHipImplicitGemmMlir::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmMlir& config) const
{
    MIOPEN_LOG_I("");
    return config.IsValidValue() && config.IsValid(ctx);
}

PerformanceImplicitGemmMlir ConvHipImplicitGemmMlir::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}

int ConvHipImplicitGemmMlir::RunAndMeasureSolution(const miopen::Handle& profile_h,
                                                   ConstData_t bot_buf,
                                                   Data_t top_buf,
                                                   ConstData_t wei_buf,
                                                   ConstData_t bias_buf,
                                                   const ConvolutionContext& ctx,
                                                   const ConvSolution& solution,
                                                   float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;

    return RunAndMeasureSolutionBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

ConvSolution ConvHipImplicitGemmMlir::GetSolution(const ConvolutionContext& ctx,
                                                  const PerformanceImplicitGemmMlir& config,
                                                  bool) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    assert(config.IsValid(ctx));

    // Adopt mlir kernel file and kernel name.

    std::string version("");
    std::string direction("");
    std::string operation("");
    std::string wrw("");
    if(ctx.direction.IsForward())
    {
        version   = "v4r4";
        operation = "conv2d";
    }
    else if(ctx.direction.IsBackwardData())
    {
        version   = "v1r1";
        direction = "backward_data_";
        operation = "conv2d_bwd_data";
    }
    else
    {
        version   = "v4r4";
        direction = "backward_weight_";
        operation = "conv2d_bwd_weight";
        wrw       = direction;
    }

    construction_parameters.kernel_file =
        "gridwise_convolution_" + wrw + "implicit_gemm_" + version + "_mlir.cpp";

    construction_parameters.kernel_name =
        "gridwise_convolution_" + direction + "implicit_gemm_" + version + "_mlir";

    // Arguments for mlir-miopen-driver.
    using CI = ConvolutionContextInterpreter;
    construction_parameters.extra_options =
    //construction_parameters.comp_options =
        std::string(" --operation ") + operation +
        std::string(" --fil_layout ") + CI::GetFilterLayout(ctx) +
        std::string(" --in_layout ") + CI::GetInputLayout(ctx) +
        std::string(" --out_layout ") + CI::GetOutputLayout(ctx) +
        std::string(" --batchsize ") + std::to_string(CI::GetBatchN(ctx)) +
        std::string(" --in_channels ") + std::to_string(CI::GetInputChannelC(ctx)) +
        std::string(" --out_channels ") + std::to_string(CI::GetOutputChannelK(ctx)) +
        std::string(" --in_h ") + std::to_string(CI::GetInputHeightHi(ctx)) +
        std::string(" --in_w ") + std::to_string(CI::GetInputWidthWi(ctx)) +
        std::string(" --out_h ") + std::to_string(CI::GetOutputHeightHo(ctx)) +
        std::string(" --out_w ") + std::to_string(CI::GetOutputWidthWo(ctx)) +
        std::string(" --fil_h ") + std::to_string(CI::GetFilterHeightY(ctx)) +
        std::string(" --fil_w ") + std::to_string(CI::GetFilterWidthX(ctx)) +
        std::string(" --dilation_h ") + std::to_string(CI::GetAdjustedConvolutionDilationH(ctx)) +
        std::string(" --dilation_w ") + std::to_string(CI::GetAdjustedConvolutionDilationW(ctx)) +
        std::string(" --conv_stride_h ") + std::to_string(CI::GetAdjustedConvolutionStrideH(ctx)) +
        std::string(" --conv_stride_w ") + std::to_string(CI::GetAdjustedConvolutionStrideW(ctx)) +
        std::string(" --padding_h ") + std::to_string(CI::GetInputLeftPadH(ctx)) +
        std::string(" --padding_w ") + std::to_string(CI::GetInputLeftPadW(ctx))
        // TBD handle left/right padding.
        // std::string(" --padding_h ") + std::string(in_right_pad_h)
        // std::string(" --padding_w ") + std::string(in_right_pad_w)
        ;
    // clang-format on

    MIOPEN_LOG_I("extra options: " << construction_parameters.extra_options);

    // construction_parameters.comp_options = ctx.general_compile_options;

    result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);
    result.construction_params.push_back(construction_parameters);
    return result;
}

} // namespace solver
} // namespace miopen