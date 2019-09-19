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
#include <array>
#include <initializer_list>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

miopenStatus_t miopenCreateTensorDescriptor(miopen::TensorDescriptor& tensorDesc)
{
    MIOPEN_LOG_FUNCTION(tensorDesc);
    return miopen::try_([&] { tensorDesc = miopen::TensorDescriptor(); });
}

miopenStatus_t miopenSet4dTensorDescriptor(
    miopen::TensorDescriptor tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, dataType, n, c, h, w);
    return miopen::try_([&] {
        std::initializer_list<int> lens = {n, c, h, w};
        tensorDesc       = miopen::TensorDescriptor(dataType, lens.begin(), 4);
    });
}

miopenStatus_t miopenGet4dTensorDescriptor(miopen::TensorDescriptor tensorDesc,
                                                      miopenDataType_t* dataType,
                                                      int* n,
                                                      int* c,
                                                      int* h,
                                                      int* w,
                                                      int* nStride,
                                                      int* cStride,
                                                      int* hStride,
                                                      int* wStride)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
    return miopen::try_([&] {
        *dataType = tensorDesc.GetType();
        miopen::tie_deref(n, c, h, w) = miopen::tien<4>(tensorDesc.GetLengths());
        miopen::tie_deref(nStride, cStride, hStride, wStride) =
            miopen::tien<4>(tensorDesc.GetStrides());
    });
}

// Internal API
// MD: This should not be required to be exported. Temporary hack
MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorLengths(
    miopen::TensorDescriptor tensorDesc, int* n, int* c, int* h, int* w)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, n, c, h, w);
    return miopen::try_([&] {
        miopen::tie_deref(n, c, h, w) = miopen::tien<4>(tensorDesc.GetLengths());
    });
}

// Internal API
MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorStrides(
    miopen::TensorDescriptor tensorDesc, int* nStride, int* cStride, int* hStride, int* wStride)
{

    //MIOPEN_LOG_FUNCTION(tensorDesc, nStride, cStride, hStride, wStride);
    return miopen::try_([&] {
        miopen::tie_deref(nStride, cStride, hStride, wStride) =
            miopen::tien<4>(tensorDesc.GetStrides());
    });
}

MIOPEN_EXPORT miopenStatus_t miopenGet5dTensorDescriptorLengths(
    miopen::TensorDescriptor tensorDesc, int* n, int* c, int* d, int* h, int* w)
{

    //MIOPEN_LOG_FUNCTION(tensorDesc, n, c, d, h, w);
    return miopen::try_([&] {
        miopen::tie_deref(n, c, d, h, w) = miopen::tien<5>(tensorDesc.GetLengths());
    });
}

// Internal API
MIOPEN_EXPORT miopenStatus_t miopenGet5dTensorDescriptorStrides(miopen::TensorDescriptor tensorDesc,
                                                                int* nStride,
                                                                int* cStride,
                                                                int* dStride,
                                                                int* hStride,
                                                                int* wStride)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, nStride, cStride, dStride, hStride, wStride);
    return miopen::try_([&] {
        miopen::tie_deref(nStride, cStride, dStride, hStride, wStride) =
            miopen::tien<5>(tensorDesc.GetStrides());
    });
}

miopenStatus_t miopenSetTensorDescriptor(miopen::TensorDescriptor tensorDesc,
                                                    miopenDataType_t dataType,
                                                    int nbDims,
                                                    int* dimsA,
                                                    int* stridesA)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, dataType, nbDims, dimsA, stridesA);
    return miopen::try_([&] {
        if(stridesA == nullptr)
        {
            tensorDesc = miopen::TensorDescriptor(dataType, dimsA, nbDims);
        }
        else
        {
            tensorDesc = miopen::TensorDescriptor(dataType, dimsA, stridesA, nbDims);
        }
    });
}

miopenStatus_t miopenGetTensorNumBytes(miopen::TensorDescriptor tensorDesc,
                                                  size_t* numBytes)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, numBytes);
    return miopen::try_([&] { *numBytes = tensorDesc.GetNumBytes(); });
}

// Internal API
int miopenGetTensorDescriptorElementSize(miopen::TensorDescriptor tensorDesc)
{
    return tensorDesc.GetElementSize();
}

miopenStatus_t miopenGetTensorDescriptorSize(miopen::TensorDescriptor tensorDesc,
                                                        int* size)
{
    MIOPEN_LOG_FUNCTION(tensorDesc, size);
    return miopen::try_([&] { *size = tensorDesc.GetSize(); });
}

miopenStatus_t miopenGetTensorDescriptor(miopen::TensorDescriptor tensorDesc,
                                                    miopenDataType_t* dataType,
                                                    int* dimsA,
                                                    int* stridesA)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, dataType, dimsA, stridesA);
    return miopen::try_([&] {
        if(dataType != nullptr)
        {
            *dataType = tensorDesc.GetType();
        }
        if(dimsA != nullptr)
        {
            std::copy(tensorDesc.GetLengths().begin(),
                      tensorDesc.GetLengths().end(),
                      dimsA);
        }
        if(stridesA != nullptr)
        {
            std::copy(tensorDesc.GetStrides().begin(),
                      tensorDesc.GetStrides().end(),
                      stridesA);
        }
    });
}

/*
miopenStatus_t miopenDestroyTensorDescriptor(miopen::TensorDescriptor tensorDesc)
{
    MIOPEN_LOG_FUNCTION(tensorDesc);
    return miopen::try_([&] { miopen_destroy_object(tensorDesc); });
}
*/

miopenStatus_t miopenOpTensor(miopen::Handle handle,
                                         miopenTensorOp_t tensorOp,
                                         const void* alpha1,
                                         const miopen::TensorDescriptor aDesc,
                                         const void* A,
                                         const void* alpha2,
                                         const miopen::TensorDescriptor bDesc,
                                         const void* B,
                                         const void* beta,
                                         const miopen::TensorDescriptor cDesc,
                                         void* C)
{

    MIOPEN_LOG_FUNCTION(tensorOp, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
    return miopen::try_([&] {
        OpTensor(handle,
                 tensorOp,
                 alpha1,
                 aDesc,
                 DataCast(A),
                 alpha2,
                 bDesc,
                 DataCast(B),
                 beta,
                 cDesc,
                 DataCast(C));
    });
}

miopenStatus_t miopenSetTensor(miopen::Handle handle,
                                          const miopen::TensorDescriptor yDesc,
                                          void* y,
                                          const void* alpha)
{

    MIOPEN_LOG_FUNCTION(handle, yDesc, y, alpha);
    return miopen::try_(
        [&] { SetTensor(handle, yDesc, DataCast(y), alpha); });
}

miopenStatus_t miopenScaleTensor(miopen::Handle handle,
                                            const miopen::TensorDescriptor yDesc,
                                            void* y,
                                            const void* alpha)
{

    MIOPEN_LOG_FUNCTION(handle, yDesc, y, alpha);
    return miopen::try_(
        [&] { ScaleTensor(handle, yDesc, DataCast(y), alpha); });
}

miopenStatus_t miopenTransformTensor(miopen::Handle handle,
                                                const void* alpha,
                                                const miopen::TensorDescriptor xDesc,
                                                const void* x,
                                                const void* beta,
                                                const miopen::TensorDescriptor yDesc,
                                                void* y)
{
    // dstValue = alpha[0]*srcValue + beta[0]*priorDstValue
    MIOPEN_LOG_FUNCTION(handle, alpha, xDesc, x, beta, yDesc, y);
    return miopen::try_([&] {
        TransformTensor(handle,
                        alpha,
                        xDesc,
                        DataCast(x),
                        beta,
                        yDesc,
                        DataCast(y));
    });
}
