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

// Internal API
// MD: This should not be required to be exported. Temporary hack
MIOPEN_EXPORT miopenStatus_t Get4dTensorDescriptorLengths(
    miopen::TensorDescriptor& tensorDesc, int* n, int* c, int* h, int* w)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, n, c, h, w);
    return miopen::try_([&] {
        miopen::tie_deref(n, c, h, w) = miopen::tien<4>(tensorDesc.GetLengths());
    });
}

MIOPEN_EXPORT miopenStatus_t Get5dTensorDescriptorLengths(
    miopen::TensorDescriptor& tensorDesc, int* n, int* c, int* d, int* h, int* w)
{

    //MIOPEN_LOG_FUNCTION(tensorDesc, n, c, d, h, w);
    return miopen::try_([&] {
        miopen::tie_deref(n, c, d, h, w) = miopen::tien<5>(tensorDesc.GetLengths());
    });
}

miopenStatus_t GetTensorDescriptorSize(miopen::TensorDescriptor& tensorDesc,
                                                        int* size)
{
    MIOPEN_LOG_FUNCTION(tensorDesc, size);
    return miopen::try_([&] { *size = tensorDesc.GetSize(); });
}

miopenStatus_t GetTensorDescriptor(miopen::TensorDescriptor& tensorDesc,
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

std::vector<int> GetTensorLengths(miopen::TensorDescriptor& tensor)
{
    int n;
    int c;
    int h;
    int w;
    int d;

    int size = 0;
    GetTensorDescriptorSize(tensor, &size);

    if(size == 5)
    {
        Get5dTensorDescriptorLengths(tensor, &n, &c, &d, &h, &w);
        return std::vector<int>({n, c, d, h, w});
    }
    else if(size == 4)
    {
        Get4dTensorDescriptorLengths(tensor, &n, &c, &h, &w);
        return std::vector<int>({n, c, h, w});
    }

    std::vector<int> tensor_len;
    tensor_len.resize(tensor.GetSize());
    GetTensorDescriptor(tensor, nullptr, tensor_len.data(), nullptr);

    return tensor_len;
}

size_t GetTensorSize(miopen::TensorDescriptor& tensor)
{
    std::vector<int> len = GetTensorLengths(tensor);
    size_t sz            = std::accumulate(len.begin(), len.end(), 1, std::multiplies<int>());

    return sz;
}
