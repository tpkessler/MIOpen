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
//#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <algorithm>

// TODO: Make miopenConvAlgoPerf_t loggable
inline std::ostream& operator<<(std::ostream& os, miopenConvAlgoPerf_t) { return os; }

miopenStatus_t
tunerConvolutionForwardGetWorkSpaceSize(miopen::Handle& handle,
                                         const miopen::TensorDescriptor& wDesc,
                                         const miopen::TensorDescriptor& xDesc,
                                         const miopen::ConvolutionDescriptor& convDesc,
                                         const miopen::TensorDescriptor& yDesc,
                                         size_t* workSpaceSize)
{

    miopen::try_([&] {
        *workSpaceSize =
            convDesc.mode == miopenTranspose
                ? convDesc.BackwardDataGetWorkSpaceSize(handle, wDesc, xDesc, yDesc)
                : convDesc.ForwardGetWorkSpaceSize(handle, wDesc, xDesc, yDesc);
    });

    return (miopenStatusSuccess);
}


miopenStatus_t
tunerFindConvolutionForwardAlgorithm(miopen::Handle& handle,
                                      const miopen::TensorDescriptor& xDesc,
                                      const void* x,
                                      const miopen::TensorDescriptor& wDesc,
                                      const void* w,
                                      const miopen::ConvolutionDescriptor& convDesc,
                                      const miopen::TensorDescriptor& yDesc,
                                      void* y,
                                      const int requestAlgoCount,
                                      int* returnedAlgoCount,
                                      miopenConvAlgoPerf_t* perfResults,
                                      void* workSpace,
                                      size_t workSpaceSize,
                                      bool exhaustiveSearch)
{

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


miopenStatus_t
tunerFindConvolutionBackwardDataAlgorithm(miopen::Handle& handle,
                                           const miopen::TensorDescriptor& dyDesc,
                                           const void* dy,
                                           const miopen::TensorDescriptor& wDesc,
                                           const void* w,
                                           const miopen::ConvolutionDescriptor& convDesc,
                                           const miopen::TensorDescriptor& dxDesc,
                                           void* dx,
                                           const int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch)
{

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
tunerConvolutionBackwardData(miopen::Handle& handle,
                              const void* alpha,
                              const miopen::TensorDescriptor& dyDesc,
                              const void* dy,
                              const miopen::TensorDescriptor& wDesc,
                              const void* w,
                              const miopen::ConvolutionDescriptor& convDesc,
                              miopenConvBwdDataAlgorithm_t algo,
                              const void* beta,
                              const miopen::TensorDescriptor& dxDesc,
                              void* dx,
                              void* workSpace,
                              size_t workSpaceSize)
{

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
tunerConvolutionBackwardDataGetWorkSpaceSize(miopen::Handle& handle,
                                              const miopen::TensorDescriptor& dyDesc,
                                              const miopen::TensorDescriptor& wDesc,
                                              const miopen::ConvolutionDescriptor& convDesc,
                                              const miopen::TensorDescriptor& dxDesc,
                                              size_t* workSpaceSize)
{

    return miopen::try_([&] {
        *workSpaceSize =
            convDesc.mode == miopenTranspose
                ? convDesc.ForwardGetWorkSpaceSize(handle, wDesc, dyDesc, dxDesc)
                : convDesc.BackwardDataGetWorkSpaceSize(handle, wDesc, dyDesc, dxDesc);
    });
}

miopenStatus_t
tunerConvolutionBackwardWeightsGetWorkSpaceSize(miopen::Handle& handle,
                                                 const miopen::TensorDescriptor& dyDesc,
                                                 const miopen::TensorDescriptor& xDesc,
                                                 const miopen::ConvolutionDescriptor& convDesc,
                                                 const miopen::TensorDescriptor& dwDesc,
                                                 size_t* workSpaceSize)
{

    return miopen::try_([&] {
        *workSpaceSize = convDesc.BackwardWeightsGetWorkSpaceSize(
            handle,
            convDesc.mode == miopenTranspose ? xDesc : dyDesc,
            convDesc.mode == miopenTranspose ? dyDesc : xDesc,
            dwDesc);
    });
}

miopenStatus_t
tunerFindConvolutionBackwardWeightsAlgorithm(miopen::Handle& handle,
                                              const miopen::TensorDescriptor& dyDesc,
                                              const void* dy,
                                              const miopen::TensorDescriptor& xDesc,
                                              const void* x,
                                              const miopen::ConvolutionDescriptor& convDesc,
                                              const miopen::TensorDescriptor& dwDesc,
                                              void* dw,
                                              const int requestAlgoCount,
                                              int* returnedAlgoCount,
                                              miopenConvAlgoPerf_t* perfResults,
                                              void* workSpace,
                                              size_t workSpaceSize,
                                              bool exhaustiveSearch)
{

    return miopen::try_([&] {
        convDesc.FindConvBwdWeightsAlgorithm(
            handle,
            /// workaround for previous trans conv logic
            convDesc.mode == miopenTranspose ? xDesc : dyDesc,
            convDesc.mode == miopenTranspose ? DataCast(x) : DataCast(dy),
            convDesc.mode == miopenTranspose ? dyDesc : xDesc,
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
tunerConvolutionBackwardWeights(miopen::Handle& handle,
                                 const void* alpha,
                                 const miopen::TensorDescriptor& dyDesc,
                                 const void* dy,
                                 const miopen::TensorDescriptor& xDesc,
                                 const void* x,
                                 const miopen::ConvolutionDescriptor& convDesc,
                                 miopenConvBwdWeightsAlgorithm_t algo,
                                 const void* beta,
                                 const miopen::TensorDescriptor& dwDesc,
                                 void* dw,
                                 void* workSpace,
                                 size_t workSpaceSize)
{

    return miopen::try_([&] {
        convDesc.ConvolutionBackwardWeights(
            handle,
            alpha,
            /// workaround for previous trans conv logic
            convDesc.mode == miopenTranspose ? xDesc : dyDesc,
            convDesc.mode == miopenTranspose ? DataCast(x) : DataCast(dy),
            convDesc.mode == miopenTranspose ? dyDesc : xDesc,
            convDesc.mode == miopenTranspose ? DataCast(dy) : DataCast(x),
            algo,
            beta,
            dwDesc,
            DataCast(dw),
            DataCast(workSpace),
            workSpaceSize);
    });
}

miopenStatus_t tunerConvolutionBackwardBias(miopen::Handle& handle,
                                                        const void* alpha,
                                                        const miopen::TensorDescriptor& dyDesc,
                                                        const void* dy,
                                                        const void* beta,
                                                        const miopen::TensorDescriptor& dbDesc,
                                                        void* db)
{
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
