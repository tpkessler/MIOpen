#ifndef CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"

namespace ck {

// This blockwise copy allow vector access of src and dst.
// It allows the vector size to be different on src and dst.
// The dimension of vector access can be different for src and dst.
// The dimension access order can be different for src and dst.
// Will do valid mapping check on src data: Read 0 if src data has a invalid mapping
// Will do valid mapping check on dst data: No write if dst data has a invalid mapping
template <index_t BlockSize,
          typename BlockSrcDesc,
          typename BlockDstDesc,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectoReadDim,
          index_t DstVectorWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace          = AddressSpace::Generic,
          AddressSpace ThreadBufferAddressSpace = AddressSpace::Generic,
          AddressSpace DstAddressSpace          = AddressSpace::Generic,
          InMemoryDataOperation DstInMemOp      = InMemoryDataOperation::Set,
          index_t NumSegments                   = 1,
          typename BlockSegmentLengths          = Sequence<1, 1>,
          typename ThreadSegmentLengths =
              typename arithmetic_sequence_gen<0, ThreadSliceLengths::Size(), 1>::type>
struct BlockwiseGenericTensorSliceCopy_v4
{
    static constexpr index_t nDim = BlockSrcDesc::GetNumOfDimension();
    using Index                   = MultiIndex<nDim>;

    __device__ constexpr BlockwiseGenericTensorSliceCopy_v4(const Index& src_block_slice_origin,
                                                            const Index& dst_block_slice_origin)
    {
        static_assert(nDim == BlockSrcDesc::GetNumOfDimension() &&
                          nDim == BlockDstDesc::GetNumOfDimension() &&
                          nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        // map threads to cluster
        constexpr auto thread_cluster_desc =
            make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

        static_assert(BlockSize == thread_cluster_desc.GetElementSize(),
                      "wrong! BlockSize not consistent with ThreadClusterLengths");

        const auto thread_cluster_id =
            thread_cluster_desc.CalculateClusterIndex(get_thread_local_1d_id());

        const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};

        mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
        mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());

        mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
        mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
    }

    struct SegmentInfo
    {
        index_t num_wave_groups;
        index_t segments_per_wave;
    };

    __device__ static constexpr auto GetSegmentInfo()
    {
        constexpr index_t long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};
        constexpr index_t block_copy_size  = BlockDstDesc::GetElementSize();

        constexpr index_t num_of_waves = BlockSize / wave_size;

        // chunk - smallest load element: a vector_load per wave
        constexpr index_t chunk_size = wave_size * long_vector_size;
        constexpr index_t num_chunks = block_copy_size / chunk_size;

        static_assert(num_chunks % NumSegments == 0,
                      "num_chunks cannot evenly divided by NumSegments");

#if 0
        constexpr index_t chunks_per_segment = num_chunks / NumSegments;

        // spread chunks to as many waves as possible
        constexpr index_t waves_per_segment = math::gcd(num_of_waves, chunks_per_segment);

        constexpr index_t num_wave_groups         = num_of_waves / waves_per_segment;
        constexpr index_t chunks_per_segment_wave = chunks_per_segment / waves_per_segment;
        constexpr index_t chunks_per_wave         = num_chunks / num_of_waves;
        constexpr index_t segments_per_wave       = chunks_per_wave / chunks_per_segment_wave;
#else
        constexpr index_t num_wave_groups   = BlockSegmentLengths::Get(Number<0>{});
        constexpr index_t segments_per_wave = BlockSegmentLengths::Get(Number<1>{});

        static_assert(num_of_waves % num_wave_groups == 0,
                      "num_of_waves cannot be divided by num_wave_groups!");

        static_assert(make_cluster_descriptor(BlockSegmentLengths{}).GetElementSize() ==
                          NumSegments,
                      "NumSegments and BlockSegmentLengths is inconsistent!");
#endif

        return SegmentInfo{num_wave_groups, segments_per_wave};
    }

    __device__ static constexpr index_t GetThreadBufferSize()
    {
        return ThreadBufferDesc::GetElementSpace();
    }

    template <typename SegmentId, typename BlockSrcData, typename ThreadBufferData>
    __device__ void RunLoadThreadBufferSegment(const BlockSrcData* p_block_src,
                                               ThreadBufferData* p_thread_buffer) const
    {
        constexpr auto seg_info     = GetSegmentInfo();
        const index_t wave_id       = get_thread_local_1d_id() / wave_size;
        const index_t wave_group_id = wave_id / seg_info.num_wave_groups;

#if 0
        const index_t active_wave_group_id = seg_id % seg_info.num_wave_groups;
        const index_t thread_seg_id             = seg_id / seg_info.num_wave_groups;
#else
        constexpr index_t active_wave_group_id = SegmentId::Get(Number<0>{});
        constexpr index_t thread_seg_id        = SegmentId::Get(Number<1>{});
#endif

        constexpr auto segment_desc = make_cluster_descriptor(ThreadSegmentLengths{});

        static_assert(ThreadSliceLengths::Size() == ThreadSegmentLengths::Size(),
                      "nDim is not consistent!");

        constexpr auto SegmentSliceLengths = ThreadSliceLengths{} / ThreadSegmentLengths{};

        static_assert(segment_desc.GetElementSize() == seg_info.segments_per_wave,
                      "ThreadSegmentLengths is wrong!");

        if(wave_group_id == active_wave_group_id)
        {
            const auto SegmentSliceOffset = segment_desc.CalculateClusterIndex(thread_seg_id);
            mThreadwiseLoad.template RunSegment<decltype(SegmentSliceLengths)>(
                p_block_src, p_thread_buffer, SegmentSliceOffset);
        }
    }

    template <typename BlockSrcData, typename ThreadBufferData>
    __device__ void RunLoadThreadBuffer(const BlockSrcData* p_block_src,
                                        ThreadBufferData* p_thread_buffer) const
    {
        constexpr bool has_optimized_address_calculation =
            decltype(mThreadwiseStore)::HasWorkingOptimizedAddressCalculation();

        // TODO: threadwise copy is still being tweaked
        if(has_optimized_address_calculation)
        {
            mThreadwiseLoad.Run_optimized_src_address_calculation(p_block_src, p_thread_buffer);
        }
        else
        {
            mThreadwiseLoad.Run(p_block_src, p_thread_buffer);
        }
    }

    template <typename ThreadBufferData, typename BlockDstData>
    __device__ void RunStoreThreadBuffer(const ThreadBufferData* p_thread_buffer,
                                         BlockDstData* p_block_dst) const
    {
        constexpr bool has_optimized_address_calculation =
            decltype(mThreadwiseStore)::HasWorkingOptimizedAddressCalculation();

        // TODO: threadwise copy is still being tweaked
        if(has_optimized_address_calculation)
        {
            mThreadwiseStore.Run_optimized_dst_address_calculation(p_thread_buffer, p_block_dst);
        }
        else
        {
            mThreadwiseStore.Run(p_thread_buffer, p_block_dst);
        }
    }

    template <typename BlockSrcData, typename BlockDstData>
    __device__ void Run(const BlockSrcData* p_block_src, BlockDstData* p_block_dst) const
    {
        static_assert(ThreadBufferAddressSpace == AddressSpace::Vgpr,
                      "wrong! This function use vgpr as its thread "
                      "buffer. However, you have set RunLoadThreadBuffer and RunStoreThreadBuffer "
                      "to use ThreadBufferAddressSpace as their thread buffer, which is not vgpr. "
                      "Behavior may be different");

        BlockSrcData p_thread_buffer[GetThreadBufferSize()];

        RunLoadThreadBuffer(p_block_src, p_thread_buffer);

        // if there is type conversion, it's done during store
        RunStoreThreadBuffer(p_thread_buffer, p_block_dst);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseLoad.MoveSrcSliceWindow(step_sizes, positive_direction);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveDstSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseStore.MoveDstSliceWindow(step_sizes, positive_direction);
    }

    private:
    using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));

    using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v4r2<BlockSrcDesc,
                                                                 ThreadBufferDesc,
                                                                 ThreadSliceLengths,
                                                                 SrcDimAccessOrder,
                                                                 SrcVectoReadDim,
                                                                 SrcDataPerRead,
                                                                 1,
                                                                 SrcAddressSpace,
                                                                 ThreadBufferAddressSpace,
                                                                 InMemoryDataOperation::Set>;

    using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v4r2<ThreadBufferDesc,
                                                                  BlockDstDesc,
                                                                  ThreadSliceLengths,
                                                                  DstDimAccessOrder,
                                                                  DstVectorWriteDim,
                                                                  1,
                                                                  DstDataPerWrite,
                                                                  ThreadBufferAddressSpace,
                                                                  DstAddressSpace,
                                                                  DstInMemOp>;

    ThreadwiseLoad mThreadwiseLoad;
    ThreadwiseStore mThreadwiseStore;

    static constexpr index_t wave_size = 64;
};

} // namespace ck

#endif
