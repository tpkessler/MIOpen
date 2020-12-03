#ifndef CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2_HPP
#define CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"

namespace ck {

// This threadwise copy allow vector access of src and dst.
// It allows the vector size to be different on src and dst.
// The dimensions of vector access should be the same on src and dst.
// The dimension access order should be the same on src and dst.
// Will do valid mapping check on src data: Read 0 if src data has a invalid mapping
// Will do valid mapping check on dst data: No write if dst data has a invalid mapping
template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorReadWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace     = AddressSpace::Generic,
          AddressSpace DstAddressSpace     = AddressSpace::Generic,
          InMemoryDataOperation DstInMemOp = InMemoryDataOperation::Set,
          index_t SrcDataStride            = 1,
          index_t DstDataStride            = 1>
struct ThreadwiseGenericTensorSliceCopy_v5
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = typename TensorCoordinate<SrcDesc>::type;
    using DstCoord = typename TensorCoordinate<DstDesc>::type;

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v5(const Index& src_slice_origin,
                                                             const Index& dst_slice_origin)
        : mSrcSliceOrigin(src_slice_origin), mDstSliceOrigin(dst_slice_origin)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::Size() &&
                          nDim == SrcDstDimAccessOrder::Size(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<SrcDstDimAccessOrder>{}, "wrong! map is not valid");

        static_assert(SliceLengths{}[SrcDstVectorReadWriteDim] %
                              math::lcm(SrcDataPerRead, DstDataPerWrite) ==
                          0,
                      "wrong! cannot evenly divide");

        // TODO:: sanity-check if vectorized memory read/write is allowed on src and dst
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v5()
        : ThreadwiseGenericTensorSliceCopy_v5(make_zero_array<index_t, nDim>(),
                                              make_zero_array<index_t, nDim>())
    {
    }

    __device__ void SetSrcSliceOrigin(SrcCoord src_slice_origin)
    {
        mSrcSliceOrigin = src_slice_origin;
    }

    __device__ void SetDstSliceOrigin(DstCoord dst_slice_origin)
    {
        mDstSliceOrigin = dst_slice_origin;
    }

    template <typename SrcData, index_t SrcDataPerAccess, index_t VectorSize>
    struct vector_data_load;

    template <>
    struct vector_data_load<float, 4, 4>
    {
        template <typename SrcCoord>
        __device__ static float_vec4_t run(const float* p_src, const SrcCoord src_coord_begin)
        {
            constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

            auto scalar_id = make_zero_array<index_t, nDim>();

            scalar_id(vector_access_dim) = 0;
            auto src_coord               = src_coord_begin + scalar_id;

            float_vec4_t r;

            r.v.s4 = load_data<float4_t, float>(p_src, src_coord.GetOffset());

            return r;
        }
    };

    template <>
    struct vector_data_load<float, 1, 4>
    {
        template <typename SrcCoord>
        __device__ static float_vec4_t run(const float* p_src, const SrcCoord src_coord_begin)
        {
            constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

            auto scalar_id = make_zero_array<index_t, nDim>();

            float_vec4_t r;

            scalar_id(vector_access_dim) = 0;
            auto src_coord               = src_coord_begin + scalar_id;
            r.v.s1.e0                    = load_data<float, float>(p_src, src_coord.GetOffset());

            scalar_id(vector_access_dim) = 1;
            src_coord                    = src_coord_begin + scalar_id;
            r.v.s1.e1                    = load_data<float, float>(p_src, src_coord.GetOffset());

            scalar_id(vector_access_dim) = 2;
            src_coord                    = src_coord_begin + scalar_id;
            r.v.s1.e2                    = load_data<float, float>(p_src, src_coord.GetOffset());

            scalar_id(vector_access_dim) = 3;
            src_coord                    = src_coord_begin + scalar_id;
            r.v.s1.e3                    = load_data<float, float>(p_src, src_coord.GetOffset());

            return r;
        }
    };

    template <>
    struct vector_data_load<float, 1, 2>
    {
        template <typename SrcCoord>
        __device__ static float_vec2_t run(const float* p_src, const SrcCoord src_coord_begin)
        {
            constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

            auto scalar_id = make_zero_array<index_t, nDim>();

            float_vec2_t r;

            scalar_id(vector_access_dim) = 0;
            auto src_coord               = src_coord_begin + scalar_id;
            r.v.s1.e0                    = load_data<float, float>(p_src, src_coord.GetOffset());

            scalar_id(vector_access_dim) = 1;
            src_coord                    = src_coord_begin + scalar_id;
            r.v.s1.e1                    = load_data<float, float>(p_src, src_coord.GetOffset());

            return r;
        }
    };

    template <typename DstData, index_t DstDataPerAccess, index_t VectorSize>
    struct vector_data_store;

    template <>
    struct vector_data_store<float, 1, 4>
    {
        template <typename DstCoord>
        __device__ static void
        run(float* p_dst, const float_vec4_t src_data, const DstCoord dst_coord_begin)
        {
            constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

            auto scalar_id = make_zero_array<index_t, nDim>();

            scalar_id(vector_access_dim) = 0;
            auto dst_coord               = dst_coord_begin + scalar_id;
            store_data<float, float>(src_data.v.s1.e0, p_dst, dst_coord.GetOffset());

            scalar_id(vector_access_dim) = 1;
            dst_coord                    = dst_coord_begin + scalar_id;
            store_data<float, float>(src_data.v.s1.e1, p_dst, dst_coord.GetOffset());

            scalar_id(vector_access_dim) = 2;
            dst_coord                    = dst_coord_begin + scalar_id;
            store_data<float, float>(src_data.v.s1.e2, p_dst, dst_coord.GetOffset());

            scalar_id(vector_access_dim) = 3;
            dst_coord                    = dst_coord_begin + scalar_id;
            store_data<float, float>(src_data.v.s1.e3, p_dst, dst_coord.GetOffset());
        }
    };

    template <>
    struct vector_data_store<float, 2, 2>
    {
        template <typename DstCoord>
        __device__ static void
        run(float* p_dst, const float_vec2_t src_data, const DstCoord dst_coord_begin)
        {
            constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

            auto scalar_id = make_zero_array<index_t, nDim>();

            scalar_id(vector_access_dim) = 0;
            auto dst_coord               = dst_coord_begin + scalar_id;
            store_data<float, float2_t>(src_data.v.s2, p_dst, dst_coord.GetOffset());
        }
    };

    template <>
    struct vector_data_store<float, 4, 4>
    {
        template <typename DstCoord>
        __device__ static void
        run(float* p_dst, const float_vec4_t src_data, const DstCoord dst_coord_begin)
        {
            constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

            auto scalar_id = make_zero_array<index_t, nDim>();

            scalar_id(vector_access_dim) = 0;
            auto dst_coord               = dst_coord_begin + scalar_id;
            store_data<float, float4_t>(src_data.v.s4, p_dst, dst_coord.GetOffset());
        }
    };

    template <typename DstData, index_t DstDataPerAccess, typename SrcData>
    struct convert_data;

    template <typename SrcData, typename DstData>
    __device__ void Run(const SrcData* p_src,
                        DstData* p_dst,
                        SrcData src_out_of_bound_value = type_convert<SrcData>{}(0.0f)) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), SrcDstDimAccessOrder>{}([&](
            auto long_vector_access_id) {

            // data id w.r.t slicing-window
            auto long_vector_data_begin_id = long_vector_access_id;
            long_vector_data_begin_id(vector_access_dim) =
                long_vector_size * long_vector_access_id[vector_access_dim];

            const auto src_coord = mSrcSliceOrigin + long_vector_data_begin_id;
            auto src_buff =
                vector_data_load<SrcData, SrcDataPerRead, long_vector_size>::run(p_src, src_coord);

            const auto dst_coord = mDstSliceOrigin + long_vector_data_begin_id;
            vector_data_store<DstData, DstDataPerWrite, long_vector_size>::run(
                p_dst, src_buff, dst_coord);
        });
    }

    template <typename SrcData, typename DstData>
    __device__ void Load(const SrcData* p_src,
                         DstData* p_dst,
                         SrcData src_out_of_bound_value = type_convert<SrcData>{}(0.0f)) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), SrcDstDimAccessOrder>{}([&](
            auto long_vector_access_id) {

            // data id w.r.t slicing-window
            auto long_vector_data_begin_id = long_vector_access_id;
            long_vector_data_begin_id(vector_access_dim) =
                long_vector_size * long_vector_access_id[vector_access_dim];

            const auto src_coord = mSrcSliceOrigin + long_vector_data_begin_id;
            auto src_buff =
                vector_data_load<SrcData, SrcDataPerRead, long_vector_size>::run(p_src, src_coord);

            const auto dst_coord = mDstSliceOrigin + long_vector_data_begin_id;
            vector_data_store<DstData, DstDataPerWrite, long_vector_size>::run(
                p_dst, src_buff, dst_coord);
        });
    }

    template <typename SrcData, typename DstData>
    __device__ void Store(const SrcData* p_src,
                          DstData* p_dst,
                          SrcData src_out_of_bound_value = type_convert<SrcData>{}(0.0f)) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), SrcDstDimAccessOrder>{}([&](
            auto long_vector_access_id) {

            // data id w.r.t slicing-window
            auto long_vector_data_begin_id = long_vector_access_id;
            long_vector_data_begin_id(vector_access_dim) =
                long_vector_size * long_vector_access_id[vector_access_dim];

            const auto src_coord = mSrcSliceOrigin + long_vector_data_begin_id;
            auto src_buff =
                vector_data_load<SrcData, SrcDataPerRead, long_vector_size>::run(p_src, src_coord);

            const auto dst_coord = mDstSliceOrigin + long_vector_data_begin_id;
            vector_data_store<DstData, DstDataPerWrite, long_vector_size>::run(
                p_dst, src_buff, dst_coord);
        });
    }

    // Modify Length to 1, if Mask is set to false
    // Used for isolating linear dimension from non-linear dimensions
    template <index_t... Lengths, index_t... Mask>
    __device__ static constexpr auto mask_lengths(Sequence<Lengths...>, Sequence<Mask...>)
    {
        return Sequence<(Mask ? Lengths : 1)...>{};
    }

    __device__ static constexpr bool HasWorkingOptimizedAddressCalculation() { return false; }

    template <typename T, bool PositiveDirection>
    __device__ void MoveSrcSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mSrcSliceOrigin += to_array(step_sizes);
        }).Else([&](auto) { mSrcSliceOrigin -= step_sizes; });
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveDstSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mDstSliceOrigin += step_sizes;
        }).Else([&](auto) { mDstSliceOrigin -= step_sizes; });
    }

    private:
    SrcCoord mSrcSliceOrigin;
    DstCoord mDstSliceOrigin;
};

} // namespace ck
#endif
