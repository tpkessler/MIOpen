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
#ifndef GUARD_MIOPEN_TUNER_HPP
#define GUARD_MIOPEN_TUNER_HPP

#include "half.hpp"
//#include "random.hpp"
#include "miopen/bfloat16.hpp"

using half_float::half;
typedef half float16;

#include "InputFlags.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/handle.hpp>
#include <numeric>
#include <vector>

#if MIOPEN_BACKEND_OPENCL

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#elif MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>

#define printf(...) fprintf(stdout, __VA_ARGS__)

#endif

#define UNPACK_VEC4(v) (v[0]), (v[1]), (v[2]), (v[3])


struct GPUMem
{

#if MIOPEN_BACKEND_OPENCL
    GPUMem(){};
    GPUMem(cl_context& ctx, size_t psz, size_t pdata_sz) : sz(psz), data_sz(pdata_sz)
    {
        buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, data_sz * sz, nullptr, nullptr);
    }

    int ToGPU(cl_command_queue& q, void* p)
    {
        return clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, data_sz * sz, p, 0, nullptr, nullptr);
    }
    int FromGPU(cl_command_queue& q, void* p)
    {
        return clEnqueueReadBuffer(q, buf, CL_TRUE, 0, data_sz * sz, p, 0, nullptr, nullptr);
    }

    cl_mem GetMem() { return buf; }
    size_t GetSize() { return sz * data_sz; }

    ~GPUMem() { clReleaseMemObject(buf); }

    cl_mem buf;
    size_t sz;
    size_t data_sz;

#elif MIOPEN_BACKEND_HIP

    GPUMem(){};
    GPUMem(uint32_t ctx, size_t psz, size_t pdata_sz) : _ctx(ctx), sz(psz), data_sz(pdata_sz)
    {
        hipMalloc(static_cast<void**>(&buf), data_sz * sz);
    }

    int ToGPU(hipStream_t q, void* p)
    {
        _q = q;
        return static_cast<int>(hipMemcpy(buf, p, data_sz * sz, hipMemcpyHostToDevice));
    }
    int FromGPU(hipStream_t q, void* p)
    {
        hipDeviceSynchronize();
        _q = q;
        return static_cast<int>(hipMemcpy(p, buf, data_sz * sz, hipMemcpyDeviceToHost));
    }

    void* GetMem() { return buf; }
    size_t GetSize() { return sz * data_sz; }

    ~GPUMem() { hipFree(buf); }
    hipStream_t _q; // Place holder for opencl context
    uint32_t _ctx;
    void* buf;
    size_t sz;
    size_t data_sz;
#endif
};

class Driver
{
    public:
    Driver()
    {
        data_type = miopenFloat;
#if MIOPEN_BACKEND_OPENCL
        //handle = miopen::Handle();
#elif MIOPEN_BACKEND_HIP
        hipStream_t s;
        hipStreamCreate(&s);
        handle = miopen::Handle(s);
#endif

        q = handle.GetStream();
    }

    miopen::Handle& GetHandle() { return handle; }
    miopenDataType_t GetDataType() { return data_type; }

#if MIOPEN_BACKEND_OPENCL
    cl_command_queue& GetStream() { return q; }
#elif MIOPEN_BACKEND_HIP
    hipStream_t& GetStream() { return q; }
#endif
    //virtual ~Driver() { miopenDestroy(handle); }

    // TODO: add timing APIs
    virtual int AddCmdLineArgs() = 0;
    virtual int ParseCmdLineArgs(int argc, char* argv[]) = 0;
    virtual InputFlags& GetInputFlags()  = 0;
    virtual int GetandSetData()          = 0;
    virtual int AllocateBuffersAndCopy() = 0;
    virtual int RunForwardGPU()          = 0;
    //virtual int VerifyForward()          = 0;
    virtual int RunBackwardGPU()         = 0;
    //virtual int VerifyBackward()         = 0;

    protected:
    template <typename Tgpu>
    void InitDataType();
    miopen::Handle handle;
    miopenDataType_t data_type;

#if MIOPEN_BACKEND_OPENCL
    cl_command_queue q;
#elif MIOPEN_BACKEND_HIP
    hipStream_t q;
#endif
};

template <>
void Driver::InitDataType<int8_t>()
{
    data_type = miopenInt8;
}
template <>
void Driver::InitDataType<float>()
{
    data_type = miopenFloat;
}
template <>
void Driver::InitDataType<float16>()
{
    data_type = miopenHalf;
}
template <>
void Driver::InitDataType<bfloat16>()
{
    data_type = miopenBFloat16;
}
// "std::is_same<Tgpu, float>{}" used to avoid "static_assert" compilation error,
// which occurs when the condition does not depend in any way on the template parameters.
template <typename Tgpu>
void Driver::InitDataType()
{
    static_assert(std::is_same<Tgpu, float>{}, "unsupported Tgpu");
}

void PadBufferSize(size_t& sz, int datatype_sz)
{
    size_t page_sz = (2 * 1024 * 1024) / datatype_sz;
    if(sz % page_sz != 0)
    {
        sz = ((sz + page_sz) / page_sz) * page_sz;
    }
}

[[gnu::noreturn]] void Usage()
{
    printf("Usage: ./MIOpenTuner *base_arg* *other_args*\n");
    printf(
        "Supported Base Arguments: conv[fp16]\n");
    exit(0);
}

std::string ParseBaseArg(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("Invalid Number of Input Arguments\n");
        Usage();
    }

    std::string arg = argv[1];

    if(arg != "conv")
    {
        printf("Invalid Base Input Argument\n");
        Usage();
    }
    else if(arg == "-h" || arg == "--help" || arg == "-?")
        Usage();
    else
        return arg;
}

class Tuner : public Driver
{
    public:
    Tuner() : Driver()
    {
        data_type = miopenFloat;
#if MIOPEN_BACKEND_OPENCL
        //handle = miopen::Handle();
#elif MIOPEN_BACKEND_HIP
        hipStream_t s;
        hipStreamCreate(&s);
        handle = miopen::Handle(s);
#endif

        q = handle.GetStream();
    }

    miopen::Handle& GetHandle() { return handle; }
    miopenDataType_t GetDataType() { return data_type; }

#if MIOPEN_BACKEND_OPENCL
    cl_command_queue& GetStream() { return q; }
#elif MIOPEN_BACKEND_HIP
    hipStream_t& GetStream() { return q; }
#endif
    //virtual ~Tuner() { miopenDestroy(handle); }

    // TODO: add timing APIs
    virtual int AddCmdLineArgs() = 0;
    virtual int ParseCmdLineArgs(int argc, char* argv[]) = 0;
    virtual InputFlags& GetInputFlags()  = 0;
    virtual int GetandSetData()          = 0;
    virtual int AllocateBuffersAndCopy() = 0;
    virtual int RunForwardGPU()          = 0;
    virtual int RunBackwardGPU()         = 0;

    protected:
    miopen::Handle handle;
    miopenDataType_t data_type;

#if MIOPEN_BACKEND_OPENCL
    cl_command_queue q;
#elif MIOPEN_BACKEND_HIP
    hipStream_t q;
#endif
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vs)
{
    os << "{ size: " << vs.size() << ", entries: ";
    for(auto& v : vs)
        os << v << " ";
    os << "}";
    return os;
}

/*! @brief Set shape of N-dimensional tensor
 *
 * Interface for querying tensor size. MIOpen has support for 1, 2, 3, 4, 5 dimensional tensor of
 * layout.
 * @param tensorDesc   Tensor descriptor type (input)
 * @param size         number of elements in tensor described by the descriptor (output)
 * @return             miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t GetTensorDescriptorSize(miopen::TensorDescriptor& tensorDesc,
                                                           int* size);

/*! @brief Get the details of the N-dimensional tensor descriptor.
 *
 * @param tensorDesc Tensor descriptor type (input)
 * @param dataType   MIOpen datatype (input)
 * @param dimsA      Array containing the size of dimensions (output)
 * @param stridesA   Array containing the size of stride (output)
 * @return           miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t GetTensorDescriptor(miopen::TensorDescriptor& tensorDesc,
                                                       miopenDataType_t* dataType,
                                                       int* dimsA,
                                                       int* stridesA);


/*! @brief Query the workspace size required for a forward convolution layer
 *
 * This call is required and must be executed once before running
 * tunerFindConvolutionForwardAlgorithm()
 * in order to determine the largest required allocation for the algorithm search; i.e., the maximum
 * size
 * of the memory needed from the set of potential forward convolution algorithm is returned.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
 *
 * @param handle         MIOpen handle (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param xDesc          Tensor descriptor for input data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param yDesc          Tensor descriptor for output data tensor y (input)
 * @param workSpaceSize  Pointer to memory to return size in bytes (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
tunerConvolutionForwardGetWorkSpaceSize(miopen::Handle& handle,
                                         const miopen::TensorDescriptor& wDesc,
                                         const miopen::TensorDescriptor& xDesc,
                                         const miopen::ConvolutionDescriptor& convDesc,
                                         const miopen::TensorDescriptor& yDesc,
                                         size_t* workSpaceSize);

/*! @brief Search and run the forward convolutional algorithms and return a list of kernel times.
 *
 * This function attempts all MIOpen forward convolution algorithms based on
 * the input configuration, and outputs performance metrics to a
 * user-allocated array of type miopenConvAlgoPerf_t. These metrics are written
 * in a sorted fashion where the first element has the lowest compute time.
 * Users can chose the top-most algorithm if they only care about the fastest
 * algorithm.
 *
 * This function is mandatory before using miopenConvolutionForward(). In order
 * to execute this function, tunerConvolutionForwardGetWorkSpaceSize() must be
 * run to determine the required memory for this search.
 *
 * * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If
 * a configuration match is not found, a default configuration will be returned.
 *
 * * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration.
 * If a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
 *
 * @param handle             MIOpen handle (input)
 * @param xDesc              Tensor descriptor for data input tensor x (input)
 * @param x                  Data tensor x (input)
 * @param wDesc              Tensor descriptor for weight tensor w (input)
 * @param w                  Weights tensor w (input)
 * @param convDesc           Convolution layer descriptor (input)
 * @param yDesc              Tensor descriptor for output data tensor y (input)
 * @param y                  Data tensor y (output)
 * @param requestAlgoCount   Number of algorithms to return kernel times (input)
 * @param returnedAlgoCount  Pointer to number of algorithms returned (output)
 * @param perfResults        Pointer to union of best algorithm for forward and backwards (input)
 * @param workSpace          Pointer to workspace required for the search (output)
 * @param workSpaceSize      Size in bytes of the memory needed for find (output)
 * @param exhaustiveSearch   A boolean to toggle a full search of all algorithms and configurations
 * (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
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
                                      bool exhaustiveSearch);

/*! @brief Get the GPU memory required for the backward data convolution algorithm.
 *
 * For a provided tensor descriptors and algorithm selection, this function calculates and returns
 * the workspace size required for back propagation on data. This call is required and must be
 * executed once before running tunerFindConvolutionBackwardDataAlgorithm() in order to determine
 * the largest required allocation for the algorithm search; i.e., the maximum size of the memory
 * needed from the set of potential backward convolution algorithm is returned.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dxDesc         Tensor descriptor for output data tensor dx (input)
 * @param workSpaceSize  Size in bytes of the memory required (output)
 * @return               miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
tunerConvolutionBackwardDataGetWorkSpaceSize(miopen::Handle& handle,
                                              const miopen::TensorDescriptor& dyDesc,
                                              const miopen::TensorDescriptor& wDesc,
                                              const miopen::ConvolutionDescriptor& convDesc,
                                              const miopen::TensorDescriptor& dxDesc,
                                              size_t* workSpaceSize);

/*! @brief Search and run the backwards data convolution algorithms and return a list of kernel
 * times.
 *
 * This function attempts all MIOpen backward data convolution algorithms, and outputs the
 * performance metrics to a user-allocated array of type miopenConvAlgoPerf_t.
 * These metrics are written in sorted fashion where the first element has the lowest compute time.
 * This function is mandatory before using backwards convolutions. Users can chose the top-most
 * algorithm if they only care about the fastest algorithm.
 *
 * This function is mandatory before using tunerConvolutionBackwardData(). In order to
 * execute this function, miopenConvolutionBackwardsDataGetWorkSpaceSize() must be run to determine
 * the required memory for this search.
 *
 * * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If
 * a configuration match is not found, a default configuration will be returned.
 *
 * * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration.
 * If a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
 *
 * @param handle             MIOpen handle (input)
 * @param dyDesc             Tensor descriptor for data input tensor dy (input)
 * @param dy                 Data delta tensor dy (input)
 * @param wDesc              Tensor descriptor for weight tensor w (input)
 * @param w                  Weights tensor w (input)
 * @param convDesc           Convolution layer descriptor (input)
 * @param dxDesc             Tensor descriptor for output data tensor dx (input)
 * @param dx                 Data delta tensor dx (input)
 * @param requestAlgoCount   Number of algorithms to return kernel times (input)
 * @param returnedAlgoCount  Pointer to number of algorithms returned (output)
 * @param perfResults        Pointer to union of best algorithm for forward and backwards (output)
 * @param workSpace          Pointer to workspace required for the search (output)
 * @param workSpaceSize      Size in bytes of the memory needed for find (output)
 * @param exhaustiveSearch   A boolean to toggle a full search of all algorithms and configurations
 * (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
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
                                           bool exhaustiveSearch);

/*! @brief Execute a backward data convolution layer
 *
 * Runs the backward data convolution layer based on the selected algorithm. The function
 * tunerFindConvolutionBackwardDataAlgorithm() must have been executed previously to
 * determine the required memory needed for the workspace and the best convolutional
 * algorithm.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param wDesc          Tensor descriptor for weight tensor w (input)
 * @param w              Weights tensor w (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param algo           Algorithm selected (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dxDesc         Tensor descriptor for output data tensor dx (input)
 * @param dx             Data delta tensor dx (output)
 * @param workSpace      Pointer to workspace required for the search (input)
 * @param workSpaceSize  Size in bytes of the memory needed for find (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
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
                              size_t workSpaceSize);

/*! @brief Get the GPU memory required for the backward weights convolution algorithm.
 *
 *
 * For a provided tensor descriptors and algorithm selection, this function calculates and returns
 * the workspace size required for back propagation on data. This call is required and must be
 * executed once before running tunerFindConvolutionBackwardWeightsAlgorithm() in order to
 * determine
 * the largest required allocation for the algorithm search; i.e., the maximum size of the memory
 * needed from the set of potential backward weights convolution algorithm is returned.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
 *
 * @param handle         MIOpen handle (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param xDesc          Tensor descriptor for data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param dwDesc         Tensor descriptor for output weights tensor dw (input)
 * @param workSpaceSize  Size in bytes of the memory required (output)
 * @return               miopenStatus_t
*/
MIOPEN_EXPORT miopenStatus_t
tunerConvolutionBackwardWeightsGetWorkSpaceSize(miopen::Handle& handle,
                                                 const miopen::TensorDescriptor& dyDesc,
                                                 const miopen::TensorDescriptor& xDesc,
                                                 const miopen::ConvolutionDescriptor& convDesc,
                                                 const miopen::TensorDescriptor& dwDesc,
                                                 size_t* workSpaceSize);

/*! @brief Search and run the backwards weights convolutional algorithms and return a list of kernel
 * times.
 *
 * This function attempts all MIOpen backward weights convolution algorithms, and outputs
 * the performance metrics to a user-allocated array of type miopenConvAlgoPerf_t. These metrics are
 * written in sorted fashion where the first element has the lowest compute time.
 * This function is mandatory before using backwards weight convolutions. Users can chose the
 * top-most algorithm if they only care about the fastest algorithm.
 *
 * This function is mandatory before using tunerConvolutionBackwardWeights(). In order to
 * execute this function, miopenConvolutionBackwardsWeightsGetWorkSpaceSize() must be run to
 * determine the required memory for this search.
 *
 * * If exhaustiveSearch == 0, MIOpen will look for the first kernel with a configuration match. If
 * a configuration match is not found, a default configuration will be returned.
 *
 * * If exhaustiveSearch == 1, MIOpen will look for the best kernel for the provided configuration.
 * If a match is not found, an exhaustive search is performed by running individual algorithms.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
 *
 * @param handle             MIOpen handle (input)
 * @param dyDesc             Tensor descriptor for data input tensor dy (input)
 * @param dy                 Data delta tensor dy (input)
 * @param xDesc              Tensor descriptor for output data tensor x (input)
 * @param x                  Data delta tensor dx (input)
 * @param convDesc           Convolution layer descriptor (input)
 * @param dwDesc             Tensor descriptor for weight tensor dw (input)
 * @param dw                 Weights delta tensor dw (input)
 * @param requestAlgoCount   Number of algorithms to return kernel times (input)
 * @param returnedAlgoCount  Pointer to number of algorithms returned (output)
 * @param perfResults        Pointer to union of best algorithm for forward and backwards (output)
 * @param workSpace          Pointer to workspace required for the search (output)
 * @param workSpaceSize      Size in bytes of the memory needed for find (output)
 * @param exhaustiveSearch   A boolean to toggle a full search of all algorithms and configurations
 * (input)
 * @return                   miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
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
                                              bool exhaustiveSearch);

/*! @brief Execute a backward weights convolution layer
 *
 * Runs the backward weights convolution layer based on the selected algorithm. The function
 * tunerFindConvolutionBackwardWeightsAlgorithm() must have
 * been executed previously to determine the required memory needed for the workspace and the
 * best convolutional algorithm.
 *
 * If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
 * this.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param dyDesc         Tensor descriptor for data tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param xDesc          Tensor descriptor for data tensor x (input)
 * @param x              Data tensor x (input)
 * @param convDesc       Convolution layer descriptor (input)
 * @param algo           Algorithm selected (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dwDesc         Tensor descriptor for weight tensor dw (input)
 * @param dw             Weights delta tensor dw (output)
 * @param workSpace      Pointer to workspace required for the search (input)
 * @param workSpaceSize  Size in bytes of the memory needed for find (input)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t
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
                                 size_t workSpaceSize);

/*! @brief Calculates the gradient with respect to the bias.
 *
 * Compute the convolution backwards gradient with respect to the bias tensor.
 *
 * @param handle         MIOpen handle (input)
 * @param alpha          Floating point scaling factor, allocated on the host (input)
 * @param dyDesc         Tensor descriptor for data input tensor dy (input)
 * @param dy             Data delta tensor dy (input)
 * @param beta           Floating point shift factor, allocated on the host (input)
 * @param dbDesc         Tensor descriptor for input bias tensor db (input)
 * @param db             Bias delta tensor db (output)
 * @return               miopenStatus_t
 */
MIOPEN_EXPORT miopenStatus_t tunerConvolutionBackwardBias(miopen::Handle& handle,
                                                           const void* alpha,
                                                           const miopen::TensorDescriptor& dyDesc,
                                                           const void* dy,
                                                           const void* beta,
                                                           const miopen::TensorDescriptor& dbDesc,
                                                           void* db);


MIOPEN_EXPORT miopenStatus_t Get4dTensorDescriptorLengths(
    miopen::TensorDescriptor& tensorDesc, int* n, int* c, int* h, int* w);

MIOPEN_EXPORT miopenStatus_t Get5dTensorDescriptorLengths(
    miopen::TensorDescriptor& tensorDesc, int* n, int* c, int* d, int* h, int* w);

std::vector<int> GetTensorLengths(miopen::TensorDescriptor& tensor);

size_t GetTensorSize(miopen::TensorDescriptor& tensor);

#endif // GUARD_MIOPEN_TUNER_HPP

