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
#ifndef GUARD_MIOPEN_TENSOR_DRIVER_HPP
#define GUARD_MIOPEN_TENSOR_DRIVER_HPP

#include <algorithm>
//#include <miopen/miopen.h>
#include "miopen.hpp"
#include <miopen/tensor.hpp>
#include <miopen/tensor_extra.hpp>
#include <numeric>
#include <vector>

MIOPEN_EXPORT int miopenGetTensorIndex(miopen::TensorDescriptor& tensorDesc,
                                       std::initializer_list<int> indices);

int miopenGetTensorDescriptorElementSize(miopen::TensorDescriptor& tensorDesc);

MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorLengths(
    miopen::TensorDescriptor& tensorDesc, int* n, int* c, int* h, int* w);

MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorStrides(
    miopen::TensorDescriptor& tensorDesc, int* nStride, int* cStride, int* hStride, int* wStride);

MIOPEN_EXPORT miopenStatus_t miopenGet5dTensorDescriptorLengths(
    miopen::TensorDescriptor& tensorDesc, int* n, int* c, int* d, int* h, int* w);

MIOPEN_EXPORT miopenStatus_t miopenGet5dTensorDescriptorStrides(miopen::TensorDescriptor& tensorDesc,
                                                                int* nStride,
                                                                int* cStride,
                                                                int* dStride,
                                                                int* hStride,
                                                                int* wStride);

std::vector<int> GetTensorLengths(miopen::TensorDescriptor& tensor)
{
    int n;
    int c;
    int h;
    int w;
    int d;

    int size = 0;
    miopenGetTensorDescriptorSize(tensor, &size);

    if(size == 5)
    {
        miopenGet5dTensorDescriptorLengths(tensor, &n, &c, &d, &h, &w);
        return std::vector<int>({n, c, d, h, w});
    }
    else if(size == 4)
    {
        miopenGet4dTensorDescriptorLengths(tensor, &n, &c, &h, &w);
        return std::vector<int>({n, c, h, w});
    }

    std::vector<int> tensor_len;
    tensor_len.resize(tensor.GetSize());
    miopenGetTensorDescriptor(tensor, nullptr, tensor_len.data(), nullptr);

    return tensor_len;
}

std::vector<int> GetTensorStrides(miopen::TensorDescriptor& tensor)
{
    int nstride;
    int cstride;
    int dstride;
    int hstride;
    int wstride;

    int size = 0;
    miopenGetTensorDescriptorSize(tensor, &size);

    if(size == 5)
    {
        miopenGet5dTensorDescriptorStrides(
            tensor, &nstride, &cstride, &dstride, &hstride, &wstride);
        return std::vector<int>({nstride, cstride, dstride, hstride, wstride});
    }
    else
    {
        miopenGet4dTensorDescriptorStrides(tensor, &nstride, &cstride, &hstride, &wstride);
        return std::vector<int>({nstride, cstride, hstride, wstride});
    }
}

int SetTensor4d(miopen::TensorDescriptor& t,
                std::vector<int>& len,
                miopenDataType_t data_type = miopenFloat)
{
    return miopenSet4dTensorDescriptor(t, data_type, UNPACK_VEC4(len));
}

int SetTensorNd(miopen::TensorDescriptor& t,
                std::vector<int>& len,
                miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), nullptr);
}

size_t GetTensorSize(miopen::TensorDescriptor& tensor)
{
    std::vector<int> len = GetTensorLengths(tensor);
    size_t sz            = std::accumulate(len.begin(), len.end(), 1, std::multiplies<int>());

    return sz;
}
#endif // GUARD_MIOPEN_TENSOR_DRIVER_HPP
