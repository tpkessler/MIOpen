/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/convolution.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <algorithm>

// TODO: Make miopenConvAlgoPerf_t loggable
inline std::ostream& operator<<(std::ostream& os, miopenConvAlgoPerf_t) { return os; }

miopenStatus_t miopenCreateConvolutionDescriptor(miopen::ConvolutionDescriptor convDesc)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_([&] { convDesc = miopen::ConvolutionDescriptor(); });
}

miopenStatus_t miopenInitConvolutionDescriptor(miopen::ConvolutionDescriptor convDesc,
                                                          miopenConvolutionMode_t c_mode,
                                                          int pad_h,
                                                          int pad_w,
                                                          int stride_h,
                                                          int stride_w,
                                                          int dilation_h,
                                                          int dilation_w)
{
    MIOPEN_LOG_FUNCTION(convDesc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    return miopen::try_([&] {
        convDesc = miopen::ConvolutionDescriptor(2,
                                                                c_mode,
                                                                miopenPaddingDefault,
                                                                {pad_h, pad_w},
                                                                {stride_h, stride_w},
                                                                {dilation_h, dilation_w});
    });
}

miopenStatus_t miopenInitConvolutionNdDescriptor(miopen::ConvolutionDescriptor convDesc,
                                                            int spatialDim,
                                                            int* padA,
                                                            int* strideA,
                                                            int* dilationA,
                                                            miopenConvolutionMode_t c_mode)
{
    MIOPEN_LOG_FUNCTION(convDesc, spatialDim, padA, strideA, dilationA, c_mode);
    return miopen::try_([&] {
        convDesc =
            miopen::ConvolutionDescriptor(spatialDim,
                                          c_mode,
                                          miopenPaddingDefault,
                                          std::vector<int>(padA, padA + spatialDim),
                                          std::vector<int>(strideA, strideA + spatialDim),
                                          std::vector<int>(dilationA, dilationA + spatialDim),
                                          std::vector<int>(spatialDim, 0),
                                          1,
                                          1.0);
    });
}

miopenStatus_t miopenSetConvolutionGroupCount(miopen::ConvolutionDescriptor convDesc,
                                                         int groupCount)
{
    MIOPEN_LOG_FUNCTION(convDesc, groupCount);
    return miopen::try_([&] { convDesc.group_count = groupCount; });
}

miopenStatus_t
miopenSetTransposeConvOutputPadding(miopen::ConvolutionDescriptor convDesc, int adj_h, int adj_w)
{
    MIOPEN_LOG_FUNCTION(convDesc, adj_h, adj_w);
    return miopen::try_([&] {
        if(convDesc.GetSpatialDimension() != 2)
        {
            MIOPEN_THROW("this API only deals with 2-D convolution");
        }

        convDesc.trans_output_pads[0] = adj_h;
        convDesc.trans_output_pads[1] = adj_w;
    });
}

miopenStatus_t miopenSetTransposeConvNdOutputPadding(
    miopen::ConvolutionDescriptor convDesc, int spatialDim, int* adjA)
{
    MIOPEN_LOG_FUNCTION(convDesc, spatialDim, adjA);
    return miopen::try_([&] {
        if(spatialDim != convDesc.GetSpatialDimension())
        {
            MIOPEN_THROW("spatialDim not consistent with convolution descriptor");
        }

        std::copy_n(adjA, spatialDim, convDesc.trans_output_pads.begin());
    });
}

miopenStatus_t miopenGetConvolutionDescriptor(miopen::ConvolutionDescriptor convDesc,
                                                         miopenConvolutionMode_t* c_mode,
                                                         int* pad_h,
                                                         int* pad_w,
                                                         int* stride_h,
                                                         int* stride_w,
                                                         int* dilation_h,
                                                         int* dilation_w)
{
    MIOPEN_LOG_FUNCTION(convDesc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    return miopen::try_([&] {
        if(convDesc.GetSpatialDimension() != 2)
        {
            MIOPEN_THROW("this API only deals with 2-D convolution");
        }

        *c_mode     = convDesc.mode;
        *pad_h      = convDesc.GetConvPads()[0];
        *pad_w      = convDesc.GetConvPads()[1];
        *stride_h   = convDesc.GetConvStrides()[0];
        *stride_w   = convDesc.GetConvStrides()[1];
        *dilation_h = convDesc.GetConvDilations()[0];
        *dilation_w = convDesc.GetConvDilations()[1];
    });
}

miopenStatus_t miopenGetConvolutionNdDescriptor(miopen::ConvolutionDescriptor convDesc,
                                                           int requestedSpatialDim,
                                                           int* spatialDim,
                                                           int* padA,
                                                           int* strideA,
                                                           int* dilationA,
                                                           miopenConvolutionMode_t* c_mode)
{
    MIOPEN_LOG_FUNCTION(
        convDesc, requestedSpatialDim, spatialDim, padA, strideA, dilationA, c_mode);
    return miopen::try_([&] {
        int spatial_dim = convDesc.GetSpatialDimension();
        if(spatial_dim < requestedSpatialDim)
        {
            MIOPEN_THROW("requestedSpatialDim is larger than actual spatial dimension");
        }
        if(spatialDim != nullptr)
        {
            *spatialDim = spatial_dim;
        }
        std::copy_n(convDesc.GetConvPads().begin(), requestedSpatialDim, padA);
        std::copy_n(convDesc.GetConvStrides().begin(), requestedSpatialDim, strideA);
        std::copy_n(
            convDesc.GetConvDilations().begin(), requestedSpatialDim, dilationA);
        if(c_mode != nullptr)
        {
            *c_mode = convDesc.mode;
        }
    });
}

miopenStatus_t
miopenGetConvolutionForwardOutputDim(miopen::ConvolutionDescriptor convDesc,
                                     const miopen::TensorDescriptor inputTensorDesc,
                                     const miopen::TensorDescriptor filterDesc,
                                     int* n,
                                     int* c,
                                     int* h,
                                     int* w)
{
    MIOPEN_LOG_FUNCTION(convDesc, inputTensorDesc, filterDesc, n, c, h, w);
    return miopen::try_([&] {
        if(convDesc.GetSpatialDimension() != 2)
        {
            MIOPEN_THROW("this API only deals with 2-D convolution");
        }

        miopen::tie_deref(n, c, h, w) = miopen::tien<4>(
            convDesc
                .GetForwardOutputTensor(inputTensorDesc, filterDesc)
                .GetLengths());
    });
}

miopenStatus_t
miopenGetConvolutionNdForwardOutputDim(miopen::ConvolutionDescriptor convDesc,
                                       const miopen::TensorDescriptor inputTensorDesc,
                                       const miopen::TensorDescriptor filterDesc,
                                       int* nDim,
                                       int* outputTensorDimA)
{
    MIOPEN_LOG_FUNCTION(convDesc, inputTensorDesc, filterDesc, nDim, outputTensorDimA);
    return miopen::try_([&] {
        auto out_desc = convDesc.GetForwardOutputTensor(
            inputTensorDesc, filterDesc);

        *nDim = out_desc.GetSize();

        for(int i = 0; i < out_desc.GetSize(); ++i)
        {
            outputTensorDimA[i] = out_desc.GetLengths()[i];
        }
    });
}

/*
miopenStatus_t miopenDestroyConvolutionDescriptor(miopen::ConvolutionDescriptor convDesc)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_([&] { miopen_destroy_object(convDesc); });
}
*/

miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize(miopen::Handle& handle,
                                         const miopen::TensorDescriptor wDesc,
                                         const miopen::TensorDescriptor xDesc,
                                         const miopen::ConvolutionDescriptor convDesc,
                                         const miopen::TensorDescriptor yDesc,
                                         size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle, wDesc, yDesc, convDesc, workSpaceSize);
    miopen::try_([&] {
        *workSpaceSize =
            convDesc.mode == miopenTranspose
                ? convDesc.BackwardDataGetWorkSpaceSize(handle,
                                                                       wDesc,
                                                                       xDesc,
                                                                       yDesc)
                : convDesc.ForwardGetWorkSpaceSize(handle,
                                                                  wDesc,
                                                                  xDesc,
                                                                  yDesc);
    });

    return (miopenStatusSuccess);
}

enum class ConvDirection
{
    Fwd = 1,
    Bwd = 2,
    WrW = 4
};

static void LogCmdConvolution(const miopen::TensorDescriptor xDesc,
                              const miopen::TensorDescriptor wDesc,
                              const miopen::ConvolutionDescriptor convDesc,
                              const ConvDirection conv_dir,
                              const bool is_immediate)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(xDesc.GetType() == miopenHalf)
        {
            ss << "convfp16";
        }
        else if(xDesc.GetType() == miopenBFloat16)
        {
            ss << "convbfp16";
        }
        else if(xDesc.GetType() == miopenInt8 ||
                xDesc.GetType() == miopenInt8x4)
        {
            ss << "convint8";
        }
        else
        {
            ss << "conv";
        }
        ss << " -n " << xDesc.GetLengths()[0] // clang-format off
            << " -c " << xDesc.GetLengths()[1]
            << " -H " << xDesc.GetLengths()[2]
            << " -W " << xDesc.GetLengths()[3]
            << " -k " << wDesc.GetLengths()[0]
            << " -y " << wDesc.GetLengths()[2]
            << " -x " << wDesc.GetLengths()[3]
            << " -p " << convDesc.GetConvPads()[0]
            << " -q " << convDesc.GetConvPads()[1]
            << " -u " << convDesc.GetConvStrides()[0]
            << " -v " << convDesc.GetConvStrides()[1]
            << " -l " << convDesc.GetConvDilations()[0]
            << " -j " << convDesc.GetConvDilations()[1]
            << " -m " << (convDesc.mode == 1 ? "trans" : "conv")
            << " -g " << convDesc.group_count
            << " -F " << std::to_string(static_cast<int>(conv_dir))
            << " -t 1"; // clang-format on
        if(xDesc.GetType() == miopenInt8x4)
            ss << " -Z 1";
        if(is_immediate)
            ss << " -S 0";
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

miopenStatus_t
miopenFindConvolutionForwardAlgorithm(miopen::Handle& handle,
                                      const miopen::TensorDescriptor xDesc,
                                      const void* x,
                                      const miopen::TensorDescriptor wDesc,
                                      const void* w,
                                      const miopen::ConvolutionDescriptor convDesc,
                                      const miopen::TensorDescriptor yDesc,
                                      void* y,
                                      const int requestAlgoCount,
                                      int* returnedAlgoCount,
                                      miopenConvAlgoPerf_t* perfResults,
                                      void* workSpace,
                                      size_t workSpaceSize,
                                      bool exhaustiveSearch)
{

    MIOPEN_LOG_FUNCTION(handle,
                        xDesc,
                        x,
                        wDesc,
                        w,
                        convDesc,
                        yDesc,
                        y,
                        requestAlgoCount,
                        returnedAlgoCount,
                        perfResults,
                        workSpace,
                        workSpaceSize,
                        exhaustiveSearch);

    /// workaround for previous trans conv logic
    if(convDesc.mode == miopenTranspose)
        return miopen::try_([&] {
            convDesc.FindConvBwdDataAlgorithm(handle,
                                                             xDesc,
                                                             DataCast(x),
                                                             wDesc,
                                                             DataCast(w),
                                                             yDesc,
                                                             DataCast(y),
                                                             requestAlgoCount,
                                                             returnedAlgoCount,
                                                             perfResults,
                                                             DataCast(workSpace),
                                                             workSpaceSize,
                                                             exhaustiveSearch);

            for(int i = 0; i < *returnedAlgoCount; ++i)
            {
                // It is guaranteed that enum values are equal, see conv_algo_name.cpp
                perfResults[i].fwd_algo =
                    static_cast<miopenConvFwdAlgorithm_t>(perfResults[i].bwd_data_algo);
            }
        });

    return miopen::try_([&] {
        convDesc.FindConvFwdAlgorithm(handle,
                                                     xDesc,
                                                     DataCast(x),
                                                     wDesc,
                                                     DataCast(w),
                                                     yDesc,
                                                     DataCast(y),
                                                     requestAlgoCount,
                                                     returnedAlgoCount,
                                                     perfResults,
                                                     DataCast(workSpace),
                                                     workSpaceSize,
                                                     exhaustiveSearch);
    });
}

miopenStatus_t miopenConvolutionForward(miopen::Handle& handle,
                                                   const void* alpha,
                                                   const miopen::TensorDescriptor xDesc,
                                                   const void* x,
                                                   const miopen::TensorDescriptor wDesc,
                                                   const void* w,
                                                   const miopen::ConvolutionDescriptor convDesc,
                                                   miopenConvFwdAlgorithm_t algo,
                                                   const void* beta,
                                                   const miopen::TensorDescriptor yDesc,
                                                   void* y,
                                                   void* workSpace,
                                                   size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle,
                        alpha,
                        xDesc,
                        x,
                        wDesc,
                        w,
                        convDesc,
                        algo,
                        beta,
                        yDesc,
                        y,
                        workSpace,
                        workSpaceSize);
    LogCmdConvolution(xDesc, wDesc, convDesc, ConvDirection::Fwd, false);

    /// workaround for previous trans conv logic
    if(convDesc.mode == miopenTranspose)
        return miopen::try_([&] {
            // It is guaranteed that enum values are equal, see conv_algo_name.cpp
            const auto algo_trans = static_cast<miopenConvBwdDataAlgorithm_t>(algo);
            convDesc.ConvolutionBackwardData(handle,
                                                            alpha,
                                                            xDesc,
                                                            DataCast(x),
                                                            wDesc,
                                                            DataCast(w),
                                                            algo_trans,
                                                            beta,
                                                            yDesc,
                                                            DataCast(y),
                                                            DataCast(workSpace),
                                                            workSpaceSize);
        });

    return miopen::try_([&] {
        convDesc.ConvolutionForward(handle,
                                                   alpha,
                                                   xDesc,
                                                   DataCast(x),
                                                   wDesc,
                                                   DataCast(w),
                                                   algo,
                                                   beta,
                                                   yDesc,
                                                   DataCast(y),
                                                   DataCast(workSpace),
                                                   workSpaceSize);
    });
}

miopenStatus_t miopenConvolutionForwardBias(miopen::Handle& handle,
                                                       const void* alpha,
                                                       const miopen::TensorDescriptor bDesc,
                                                       const void* b,
                                                       const void* beta,
                                                       const miopen::TensorDescriptor yDesc,
                                                       void* y)
{

    MIOPEN_LOG_FUNCTION(handle, alpha, bDesc, b, beta, yDesc, y);

    // bfloat16 not supported for bias operation
    if(yDesc.GetType() == miopenBFloat16 ||
       bDesc.GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }

    return miopen::try_([&] {
        return OpTensor(handle,
                        miopenTensorOpAdd,
                        alpha,
                        yDesc,
                        DataCast(y),
                        alpha,
                        bDesc,
                        DataCast(b),
                        beta,
                        yDesc,
                        DataCast(y));
    });
}

miopenStatus_t
miopenConvolutionForwardGetSolutionCount(miopen::Handle& handle,
                                         const miopen::TensorDescriptor wDesc,
                                         const miopen::TensorDescriptor xDesc,
                                         const miopen::ConvolutionDescriptor convDesc,
                                         const miopen::TensorDescriptor yDesc,
                                         size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            *solutionCount = convDesc.GetBackwardSolutionCount(handle,
                                                                              xDesc,
                                                                              wDesc,
                                                                              yDesc);
        else
            *solutionCount = convDesc.GetForwardSolutionCount(handle,
                                                                             wDesc,
                                                                             xDesc,
                                                                             yDesc);
    });
}

miopenStatus_t
miopenConvolutionForwardGetSolution(miopen::Handle& handle,
                                    const miopen::TensorDescriptor wDesc,
                                    const miopen::TensorDescriptor xDesc,
                                    const miopen::ConvolutionDescriptor convDesc,
                                    const miopen::TensorDescriptor yDesc,
                                    const size_t maxSolutionCount,
                                    size_t* solutionCount,
                                    miopenConvSolution_t* solutions)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc, maxSolutionCount);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.GetBackwardSolutions(handle,
                                                         xDesc,
                                                         wDesc,
                                                         yDesc,
                                                         maxSolutionCount,
                                                         solutionCount,
                                                         solutions);
        else
            convDesc.GetForwardSolutions(handle,
                                                        wDesc,
                                                        xDesc,
                                                        yDesc,
                                                        maxSolutionCount,
                                                        solutionCount,
                                                        solutions);
    });
}

miopenStatus_t
miopenConvolutionForwardGetSolutionWorkspaceSize(miopen::Handle& handle,
                                                 const miopen::TensorDescriptor wDesc,
                                                 const miopen::TensorDescriptor xDesc,
                                                 const miopen::ConvolutionDescriptor convDesc,
                                                 const miopen::TensorDescriptor yDesc,
                                                 const uint64_t solution_id,
                                                 size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc, solution_id, workSpaceSize);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            *workSpaceSize =
                convDesc.GetBackwardSolutionWorkspaceSize(handle,
                                                                         xDesc,
                                                                         wDesc,
                                                                         yDesc,
                                                                         solution_id);
        else
            *workSpaceSize =
                convDesc.GetForwardSolutionWorkspaceSize(handle,
                                                                        wDesc,
                                                                        xDesc,
                                                                        yDesc,
                                                                        solution_id);
    });
}

miopenStatus_t
miopenConvolutionForwardCompileSolution(miopen::Handle& handle,
                                        const miopen::TensorDescriptor wDesc,
                                        const miopen::TensorDescriptor xDesc,
                                        const miopen::ConvolutionDescriptor convDesc,
                                        const miopen::TensorDescriptor yDesc,
                                        const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc, solution_id);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.CompileBackwardSolution(handle,
                                                            xDesc,
                                                            wDesc,
                                                            yDesc,
                                                            solution_id);
        else
            convDesc.CompileForwardSolution(handle,
                                                           wDesc,
                                                           xDesc,
                                                           yDesc,
                                                           solution_id);
    });
}

miopenStatus_t
miopenConvolutionForwardImmediate(miopen::Handle& handle,
                                  const miopen::TensorDescriptor wDesc,
                                  const void* w,
                                  const miopen::TensorDescriptor xDesc,
                                  const void* x,
                                  const miopen::ConvolutionDescriptor convDesc,
                                  const miopen::TensorDescriptor yDesc,
                                  void* y,
                                  void* workSpace,
                                  size_t workSpaceSize,
                                  const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(
        handle, wDesc, w, xDesc, x, convDesc, yDesc, y, workSpace, workSpaceSize, solution_id);
    LogCmdConvolution(xDesc, wDesc, convDesc, ConvDirection::Fwd, true);

    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.ConvolutionBackwardImmediate(handle,
                                                                 xDesc,
                                                                 DataCast(x),
                                                                 wDesc,
                                                                 DataCast(w),
                                                                 yDesc,
                                                                 DataCast(y),
                                                                 DataCast(workSpace),
                                                                 workSpaceSize,
                                                                 solution_id);
        else
            convDesc.ConvolutionForwardImmediate(handle,
                                                                wDesc,
                                                                DataCast(w),
                                                                xDesc,
                                                                DataCast(x),
                                                                yDesc,
                                                                DataCast(y),
                                                                DataCast(workSpace),
                                                                workSpaceSize,
                                                                solution_id);
    });
}

miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount(miopen::Handle& handle,
                                              const miopen::TensorDescriptor dyDesc,
                                              const miopen::TensorDescriptor wDesc,
                                              const miopen::ConvolutionDescriptor convDesc,
                                              const miopen::TensorDescriptor dxDesc,
                                              size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            *solutionCount = convDesc.GetForwardSolutionCount(handle,
                                                                             wDesc,
                                                                             dyDesc,
                                                                             dxDesc);
        else
            *solutionCount =
                convDesc.GetBackwardSolutionCount(handle,
                                                                 dyDesc,
                                                                 wDesc,
                                                                 dxDesc);
    });
}

miopenStatus_t
miopenConvolutionBackwardDataGetSolution(miopen::Handle& handle,
                                         const miopen::TensorDescriptor dyDesc,
                                         const miopen::TensorDescriptor wDesc,
                                         const miopen::ConvolutionDescriptor convDesc,
                                         const miopen::TensorDescriptor dxDesc,
                                         const size_t maxSolutionCount,
                                         size_t* solutionCount,
                                         miopenConvSolution_t* solutions)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, maxSolutionCount, solutionCount);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.GetForwardSolutions(handle,
                                                        wDesc,
                                                        dyDesc,
                                                        dxDesc,
                                                        maxSolutionCount,
                                                        solutionCount,
                                                        solutions);

        else
            convDesc.GetBackwardSolutions(handle,
                                                         dyDesc,
                                                         wDesc,
                                                         dxDesc,
                                                         maxSolutionCount,
                                                         solutionCount,
                                                         solutions);
    });
}

miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize(miopen::Handle& handle,
                                                      const miopen::TensorDescriptor dyDesc,
                                                      const miopen::TensorDescriptor wDesc,
                                                      const miopen::ConvolutionDescriptor convDesc,
                                                      const miopen::TensorDescriptor dxDesc,
                                                      const uint64_t solution_id,
                                                      size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, solution_id, workSpaceSize);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            *workSpaceSize =
                convDesc.GetForwardSolutionWorkspaceSize(handle,
                                                                        wDesc,
                                                                        dyDesc,
                                                                        dxDesc,
                                                                        solution_id);
        else
            *workSpaceSize =
                convDesc.GetBackwardSolutionWorkspaceSize(handle,
                                                                         dyDesc,
                                                                         wDesc,
                                                                         dxDesc,
                                                                         solution_id);
    });
}

miopenStatus_t
miopenConvolutionBackwardDataCompileSolution(miopen::Handle& handle,
                                             const miopen::TensorDescriptor dyDesc,
                                             const miopen::TensorDescriptor wDesc,
                                             const miopen::ConvolutionDescriptor convDesc,
                                             const miopen::TensorDescriptor dxDesc,
                                             const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, solution_id);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.CompileForwardSolution(handle,
                                                           wDesc,
                                                           dyDesc,
                                                           dxDesc,
                                                           solution_id);
        else
            convDesc.CompileBackwardSolution(handle,
                                                            dyDesc,
                                                            wDesc,
                                                            dxDesc,
                                                            solution_id);
    });
}

miopenStatus_t
miopenConvolutionBackwardDataImmediate(miopen::Handle& handle,
                                       const miopen::TensorDescriptor dyDesc,
                                       const void* dy,
                                       const miopen::TensorDescriptor wDesc,
                                       const void* w,
                                       const miopen::ConvolutionDescriptor convDesc,
                                       const miopen::TensorDescriptor dxDesc,
                                       void* dx,
                                       void* workSpace,
                                       size_t workSpaceSize,
                                       const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(
        handle, dyDesc, wDesc, convDesc, dxDesc, workSpace, workSpaceSize, solution_id);
    LogCmdConvolution(dxDesc, wDesc, convDesc, ConvDirection::Bwd, true);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.ConvolutionForwardImmediate(handle,
                                                                wDesc,
                                                                DataCast(w),
                                                                dyDesc,
                                                                DataCast(dy),
                                                                dxDesc,
                                                                DataCast(dx),
                                                                DataCast(workSpace),
                                                                workSpaceSize,
                                                                solution_id);
        else
            convDesc.ConvolutionBackwardImmediate(handle,
                                                                 dyDesc,
                                                                 DataCast(dy),
                                                                 wDesc,
                                                                 DataCast(w),
                                                                 dxDesc,
                                                                 DataCast(dx),
                                                                 DataCast(workSpace),
                                                                 workSpaceSize,
                                                                 solution_id);
    });
}
miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount(miopen::Handle& handle,
                                                 const miopen::TensorDescriptor dyDesc,
                                                 const miopen::TensorDescriptor xDesc,
                                                 const miopen::ConvolutionDescriptor convDesc,
                                                 const miopen::TensorDescriptor dwDesc,
                                                 size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, solutionCount);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            *solutionCount = convDesc.GetWrwSolutionCount(handle,
                                                                         xDesc,
                                                                         dyDesc,
                                                                         dwDesc);
        else
            *solutionCount = convDesc.GetWrwSolutionCount(handle,
                                                                         dyDesc,
                                                                         xDesc,
                                                                         dwDesc);
    });
}

miopenStatus_t
miopenConvolutionBackwardWeightsGetSolution(miopen::Handle& handle,
                                            const miopen::TensorDescriptor dyDesc,
                                            const miopen::TensorDescriptor xDesc,
                                            const miopen::ConvolutionDescriptor convDesc,
                                            const miopen::TensorDescriptor dwDesc,
                                            const size_t maxSolutionCount,
                                            size_t* solutionCount,
                                            miopenConvSolution_t* solutions)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, maxSolutionCount, solutionCount);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.GetWrwSolutions(handle,
                                                    xDesc,
                                                    dyDesc,
                                                    dwDesc,
                                                    maxSolutionCount,
                                                    solutionCount,
                                                    solutions);
        else
            convDesc.GetWrwSolutions(handle,
                                                    dyDesc,
                                                    xDesc,
                                                    dwDesc,
                                                    maxSolutionCount,
                                                    solutionCount,
                                                    solutions);
    });
}

miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopen::Handle& handle,
    const miopen::TensorDescriptor dyDesc,
    const miopen::TensorDescriptor xDesc,
    const miopen::ConvolutionDescriptor convDesc,
    const miopen::TensorDescriptor dwDesc,
    const uint64_t solution_id,
    size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, solution_id, workSpaceSize);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            *workSpaceSize =
                convDesc.GetWrwSolutionWorkspaceSize(handle,
                                                                    xDesc,
                                                                    dyDesc,
                                                                    dwDesc,
                                                                    solution_id);
        else
            *workSpaceSize =
                convDesc.GetWrwSolutionWorkspaceSize(handle,
                                                                    dyDesc,
                                                                    xDesc,
                                                                    dwDesc,
                                                                    solution_id);
    });
}

miopenStatus_t
miopenConvolutionBackwardWeightsCompileSolution(miopen::Handle& handle,
                                                const miopen::TensorDescriptor dyDesc,
                                                const miopen::TensorDescriptor xDesc,
                                                const miopen::ConvolutionDescriptor convDesc,
                                                const miopen::TensorDescriptor dwDesc,
                                                const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, solution_id);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.CompileWrwSolution(handle,
                                                       xDesc,
                                                       dyDesc,
                                                       dwDesc,
                                                       solution_id);
        else
            convDesc.CompileWrwSolution(handle,
                                                       dyDesc,
                                                       xDesc,
                                                       dwDesc,
                                                       solution_id);
    });
}

miopenStatus_t
miopenConvolutionBackwardWeightsImmediate(miopen::Handle& handle,
                                          const miopen::TensorDescriptor dyDesc,
                                          const void* dy,
                                          const miopen::TensorDescriptor xDesc,
                                          const void* x,
                                          const miopen::ConvolutionDescriptor convDesc,
                                          const miopen::TensorDescriptor dwDesc,
                                          void* dw,
                                          void* workSpace,
                                          size_t workSpaceSize,
                                          const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(
        handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, workSpace, workSpaceSize, solution_id);
    LogCmdConvolution(xDesc, dwDesc, convDesc, ConvDirection::WrW, true);
    return miopen::try_([&] {
        if(convDesc.mode == miopenTranspose)
            convDesc.ConvolutionWrwImmediate(handle,
                                                            xDesc,
                                                            DataCast(x),
                                                            dyDesc,
                                                            DataCast(dy),
                                                            dwDesc,
                                                            DataCast(dw),
                                                            DataCast(workSpace),
                                                            workSpaceSize,
                                                            solution_id);
        else
            convDesc.ConvolutionWrwImmediate(handle,
                                                            dyDesc,
                                                            DataCast(dy),
                                                            xDesc,
                                                            DataCast(x),
                                                            dwDesc,
                                                            DataCast(dw),
                                                            DataCast(workSpace),
                                                            workSpaceSize,
                                                            solution_id);
    });
}

miopenStatus_t
miopenFindConvolutionBackwardDataAlgorithm(miopen::Handle& handle,
                                           const miopen::TensorDescriptor dyDesc,
                                           const void* dy,
                                           const miopen::TensorDescriptor wDesc,
                                           const void* w,
                                           const miopen::ConvolutionDescriptor convDesc,
                                           const miopen::TensorDescriptor dxDesc,
                                           void* dx,
                                           const int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch)
{

    MIOPEN_LOG_FUNCTION(handle,
                        dyDesc,
                        dy,
                        wDesc,
                        w,
                        convDesc,
                        dxDesc,
                        dx,
                        requestAlgoCount,
                        returnedAlgoCount,
                        perfResults,
                        workSpace,
                        workSpaceSize,
                        exhaustiveSearch);

    /// workaround for previous trans conv logic
    if(convDesc.mode == miopenTranspose)
        return miopen::try_([&] {
            convDesc.FindConvFwdAlgorithm(handle,
                                                         dyDesc,
                                                         DataCast(dy),
                                                         wDesc,
                                                         DataCast(w),
                                                         dxDesc,
                                                         DataCast(dx),
                                                         requestAlgoCount,
                                                         returnedAlgoCount,
                                                         perfResults,
                                                         DataCast(workSpace),
                                                         workSpaceSize,
                                                         exhaustiveSearch);

            for(int i = 0; i < *returnedAlgoCount; ++i)
            {
                // It is guaranteed that enum values are equal, see conv_algo_name.cpp
                perfResults[i].bwd_data_algo =
                    static_cast<miopenConvBwdDataAlgorithm_t>(perfResults[i].fwd_algo);
            }
        });

    return miopen::try_([&] {
        convDesc.FindConvBwdDataAlgorithm(handle,
                                                         dyDesc,
                                                         DataCast(dy),
                                                         wDesc,
                                                         DataCast(w),
                                                         dxDesc,
                                                         DataCast(dx),
                                                         requestAlgoCount,
                                                         returnedAlgoCount,
                                                         perfResults,
                                                         DataCast(workSpace),
                                                         workSpaceSize,
                                                         exhaustiveSearch);
    });
}

miopenStatus_t
miopenConvolutionBackwardData(miopen::Handle& handle,
                              const void* alpha,
                              const miopen::TensorDescriptor dyDesc,
                              const void* dy,
                              const miopen::TensorDescriptor wDesc,
                              const void* w,
                              const miopen::ConvolutionDescriptor convDesc,
                              miopenConvBwdDataAlgorithm_t algo,
                              const void* beta,
                              const miopen::TensorDescriptor dxDesc,
                              void* dx,
                              void* workSpace,
                              size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle,
                        alpha,
                        dyDesc,
                        dy,
                        wDesc,
                        w,
                        convDesc,
                        algo,
                        beta,
                        dxDesc,
                        dx,
                        workSpace,
                        workSpaceSize);
    LogCmdConvolution(dxDesc, wDesc, convDesc, ConvDirection::Bwd, false);

    /// workaround for previous trans conv logic
    if(convDesc.mode == miopenTranspose)
        return miopen::try_([&] {
            // It is guaranteed that enum values are equal, see conv_algo_name.cpp
            const auto algo_trans = static_cast<miopenConvFwdAlgorithm_t>(algo);
            convDesc.ConvolutionForward(handle,
                                                       alpha,
                                                       dyDesc,
                                                       DataCast(dy),
                                                       wDesc,
                                                       DataCast(w),
                                                       algo_trans,
                                                       beta,
                                                       dxDesc,
                                                       DataCast(dx),
                                                       DataCast(workSpace),
                                                       workSpaceSize);
        });

    return miopen::try_([&] {
        convDesc.ConvolutionBackwardData(handle,
                                                        alpha,
                                                        dyDesc,
                                                        DataCast(dy),
                                                        wDesc,
                                                        DataCast(w),
                                                        algo,
                                                        beta,
                                                        dxDesc,
                                                        DataCast(dx),
                                                        DataCast(workSpace),
                                                        workSpaceSize);
    });
}

miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSize(miopen::Handle& handle,
                                              const miopen::TensorDescriptor dyDesc,
                                              const miopen::TensorDescriptor wDesc,
                                              const miopen::ConvolutionDescriptor convDesc,
                                              const miopen::TensorDescriptor dxDesc,
                                              size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, workSpaceSize);
    return miopen::try_([&] {
        *workSpaceSize =
            convDesc.mode == miopenTranspose
                ? convDesc.ForwardGetWorkSpaceSize(handle,
                                                                  wDesc,
                                                                  dyDesc,
                                                                  dxDesc)
                : convDesc.BackwardDataGetWorkSpaceSize(handle,
                                                                       wDesc,
                                                                       dyDesc,
                                                                       dxDesc);
    });
}

miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopen::Handle& handle,
                                                 const miopen::TensorDescriptor dyDesc,
                                                 const miopen::TensorDescriptor xDesc,
                                                 const miopen::ConvolutionDescriptor convDesc,
                                                 const miopen::TensorDescriptor dwDesc,
                                                 size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, workSpaceSize);
    return miopen::try_([&] {
        *workSpaceSize = convDesc.BackwardWeightsGetWorkSpaceSize(
            handle,
            convDesc.mode == miopenTranspose ? xDesc
                                                            : dyDesc,
            convDesc.mode == miopenTranspose ? dyDesc
                                                            : xDesc,
            dwDesc);
    });
}

miopenStatus_t
miopenFindConvolutionBackwardWeightsAlgorithm(miopen::Handle& handle,
                                              const miopen::TensorDescriptor dyDesc,
                                              const void* dy,
                                              const miopen::TensorDescriptor xDesc,
                                              const void* x,
                                              const miopen::ConvolutionDescriptor convDesc,
                                              const miopen::TensorDescriptor dwDesc,
                                              void* dw,
                                              const int requestAlgoCount,
                                              int* returnedAlgoCount,
                                              miopenConvAlgoPerf_t* perfResults,
                                              void* workSpace,
                                              size_t workSpaceSize,
                                              bool exhaustiveSearch)
{

    MIOPEN_LOG_FUNCTION(handle,
                        dyDesc,
                        dy,
                        xDesc,
                        x,
                        convDesc,
                        dwDesc,
                        dw,
                        requestAlgoCount,
                        returnedAlgoCount,
                        perfResults,
                        workSpace,
                        workSpaceSize,
                        exhaustiveSearch);
    LogCmdConvolution(xDesc, dwDesc, convDesc, ConvDirection::WrW, false);

    return miopen::try_([&] {
        convDesc.FindConvBwdWeightsAlgorithm(
            handle,
            /// workaround for previous trans conv logic
            convDesc.mode == miopenTranspose ? xDesc
                                                            : dyDesc,
            convDesc.mode == miopenTranspose ? DataCast(x) : DataCast(dy),
            convDesc.mode == miopenTranspose ? dyDesc
                                                            : xDesc,
            convDesc.mode == miopenTranspose ? DataCast(dy) : DataCast(x),
            dwDesc,
            DataCast(dw),
            requestAlgoCount,
            returnedAlgoCount,
            perfResults,
            DataCast(workSpace),
            workSpaceSize,
            exhaustiveSearch);
    });
}

miopenStatus_t
miopenConvolutionBackwardWeights(miopen::Handle& handle,
                                 const void* alpha,
                                 const miopen::TensorDescriptor dyDesc,
                                 const void* dy,
                                 const miopen::TensorDescriptor xDesc,
                                 const void* x,
                                 const miopen::ConvolutionDescriptor convDesc,
                                 miopenConvBwdWeightsAlgorithm_t algo,
                                 const void* beta,
                                 const miopen::TensorDescriptor dwDesc,
                                 void* dw,
                                 void* workSpace,
                                 size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle,
                        alpha,
                        dyDesc,
                        dy,
                        xDesc,
                        x,
                        convDesc,
                        algo,
                        beta,
                        dwDesc,
                        dw,
                        workSpace,
                        workSpaceSize);
    return miopen::try_([&] {
        convDesc.ConvolutionBackwardWeights(
            handle,
            alpha,
            /// workaround for previous trans conv logic
            convDesc.mode == miopenTranspose ? xDesc
                                                            : dyDesc,
            convDesc.mode == miopenTranspose ? DataCast(x) : DataCast(dy),
            convDesc.mode == miopenTranspose ? dyDesc
                                                            : xDesc,
            convDesc.mode == miopenTranspose ? DataCast(dy) : DataCast(x),
            algo,
            beta,
            dwDesc,
            DataCast(dw),
            DataCast(workSpace),
            workSpaceSize);
    });
}

miopenStatus_t miopenConvolutionBackwardBias(miopen::Handle& handle,
                                                        const void* alpha,
                                                        const miopen::TensorDescriptor dyDesc,
                                                        const void* dy,
                                                        const void* beta,
                                                        const miopen::TensorDescriptor dbDesc,
                                                        void* db)
{
    MIOPEN_LOG_FUNCTION(handle, alpha, dyDesc, dy, beta, dbDesc, db);
    // bfloat16 not supported for bias operation
    if(dyDesc.GetType() == miopenBFloat16 ||
       dbDesc.GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }

    return miopen::try_([&] {
        ConvolutionBackwardBias(handle,
                                alpha,
                                dyDesc,
                                DataCast(dy),
                                beta,
                                dbDesc,
                                DataCast(db));
    });
}
