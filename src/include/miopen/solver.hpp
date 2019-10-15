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

#ifndef GUARD_MIOPEN_SOLVER_HPP_
#define GUARD_MIOPEN_SOLVER_HPP_

#include <miopen/config.h>

#include <miopen/buffer_info.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/performance_config.hpp>
#include <miopen/scgemm_param.hpp>
#include <miopen/type_name.hpp>

#include <memory>
#include <string>
#include <vector>
#include <ostream>
#include <algorithm>

// Todo: remove
#include <miopen/db.hpp>

namespace miopen {

namespace solver {
/// \todo Move wave_size into abstraction wich represent GPU information
const int wave_size = 64;
/// Base class for problem solvers.
///
/// Solvers are to be instantiated as const objects and shall not have any variable
/// internal state. Any non-const state information, if required, to be stored in the
/// solver-specific context objects.
///
/// There could be multiple solvers of the same algorithm for a problem config.
/// For example, ConvAsm3x3U and ConvOclDirectFwd3x3
/// are able to solve overlapping sets of 3x3 Direct convolution problems.
template <class TContext>
struct SolverBase
{
    SolverBase(const SolverBase&) {}
    SolverBase& operator  =(const SolverBase&) { return *this; }
    SolverBase()          = default;
    virtual ~SolverBase() = default;

    /// Returns true if solution can work on given SW/HW platform (runtime/device)
    /// and provides correct result for the problem config.
    ///
    /// Every SolverBase which IsApplicable() for some problem config must be able to
    /// GetPerformanceConfig() so that GetSolution() would return valid
    /// solution for a problem (i.e. convolution). In other words, if a Solution
    /// says "I'm suitable" for a problem, it agrees to solve that problem correctly.
    virtual bool IsApplicable(const TContext&) const = 0;

    /// Legacy euristic method which shall return false when a solution
    /// is known to be slower than some another solution for the same problem config.
    /// Intended to be used for performance optimization.
    /// Warning: Non-trivial implementations introduce implicit dependencies between solutions.
    virtual bool IsFast(const TContext&) const { return true; }
    // Returns the workspace size required by the solver for a given ConvolutionContext
    virtual size_t GetWorkspaceSize(const TContext&) const { return 0; };

    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    virtual ConvSolution GetSolution(const TContext& params) const = 0;

    /// Temporary solver-specific method until we have generic means for running solutions.
    /// int RunAndMeasureSolution(miopen::Handle& profile_h,
    ///                          Data_t bot_ocl_buf,
    ///                          Data_t top_ocl_buf,
    ///                          Data_t wei_ocl_buf,
    ///                          Data_t bias_ocl_buf,
    ///                          const TContext& params,
    ///                          const ConvSolution& solution,
    ///                          float& elapsed_time) const;

    virtual const std::string& DbId() const = 0;

    protected:
    template <class Solver>
    static const std::string& SolverDbId(const Solver& solver)
    {
        static const auto result = ComputeSolverDbId(solver);
        return result;
    }

    private:
    template <class Solver>
    static std::string ComputeSolverDbId(const Solver&)
    {
        const auto& const_name = get_type_name<Solver>();
        auto idx               = const_name.find_last_of(':');
        auto name              = const_name.substr(idx + 1);
        std::replace(name.begin(), name.end(), ',', '-');
        name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
        return name;
    }
};

/// Base class for problem solvers which use exhaustive search mechanism.
template <class TContext>
struct SearchableSolver : virtual SolverBase<TContext>
{
    /// Initializes performance config to the default values.
    /// The function may involve some euristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    virtual std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const TContext&) const = 0;

    virtual ConvSolution GetSolution(const TContext& params,
                                     const IPerformanceConfig& config,
                                     bool disableConfigOverrideFromEnv) const = 0;

    protected:
    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfig(const TContext&, const IPerformanceConfig&) const
    {
        return true; // Do not check by default.
    }

    private:
    virtual std::shared_ptr<IPerformanceConfig> Search(const TContext&) const = 0;

    ConvSolution GetSolution(const TContext& context) const final
    {
        const auto& db_id = this->DbId();
        if(context.disable_perfdb_access)
        {
            MIOPEN_LOG_I(db_id << " (db access disabled)");
            return GetSolution(context, *GetPerformanceConfig(context), false);
        }
        auto db = GetDb(context);
        const FindEnforce enforce;
        MIOPEN_LOG_I(db_id);
        if(enforce.IsDbClean(context))
        {
            if(db.Remove(context, db_id))
                MIOPEN_LOG_W("Perf Db: record removed: " << db_id << ", enforce: " << enforce);
        }
        else
        {
            if((context.do_search || enforce.IsSearch(context)) && enforce.IsDbUpdate(context))
            {
                MIOPEN_LOG_W("Perf Db: load skipped: " << db_id << ", enforce: " << enforce);
            }
            else
            {
                std::shared_ptr<IPerformanceConfig> config;
                if(db.Load(context, db_id, *config))
                {
                    MIOPEN_LOG_I2("Perf Db: record loaded: " << db_id);
                    if(this->IsValidPerformanceConfig(context, *config))
                    {
                        return GetSolution(context, *config, false);
                    }
                    MIOPEN_LOG(
                        (MIOPEN_INSTALLABLE ? LoggingLevel::Warning : miopen::LoggingLevel::Error),
                        "Invalid config loaded from Perf Db: " << db_id << ": " << config
                                                               << ". Performance may degrade.");
                }
                else
                {
                    MIOPEN_LOG_I("Perf Db: record not found for: " << db_id);
                }
            }

            if(context.do_search ||
               enforce.IsSearch(context)) // TODO: Make it a customization point
            {
                MIOPEN_LOG_I("Starting search: " << db_id << ", enforce: " << enforce);
                try
                {
                    auto c = Search(context);
                    db.Update(context, db_id, *c);
                    return GetSolution(context, *c, false);
                }
                catch(const miopen::Exception& ex)
                {
                    MIOPEN_LOG_E("Search failed for: " << db_id << ": " << ex.what());
                }
            }
        }

        return GetSolution(context, *GetPerformanceConfig(context), false);
    }
};

template <class TContext>
struct GenericSearchableSolver : virtual SearchableSolver<TContext>
{
    /// Initialize performance config for the generic search.
    virtual std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const = 0;

    virtual int RunAndMeasureSolutionFwd(miopen::Handle& /*profile_h*/,
                                         ConstData_t /*bot_ocl_buf*/,
                                         Data_t /*top_ocl_buf*/,
                                         ConstData_t /*wei_ocl_buf*/,
                                         ConstData_t /*bias_ocl_buf*/,
                                         const ConvolutionContext& /*params*/,
                                         const ConvSolution& /*solution*/,
                                         float& /*elapsed_time*/) const
    {
        MIOPEN_THROW("Not implemented");
    }

    virtual int RunAndMeasureSolutionBwd(miopen::Handle& /*profile_h*/,
                                         Data_t /*bot_ocl_buf*/,
                                         ConstData_t /*top_ocl_buf*/,
                                         ConstData_t /*wei_ocl_buf*/,
                                         ConstData_t /*bias_ocl_buf*/,
                                         const ConvolutionContext& /*params*/,
                                         const ConvSolution& /*solution*/,
                                         float& /*elapsed_time*/) const
    {
        MIOPEN_THROW("Not implemented");
    }

    virtual int RunAndMeasureSolutionWrW(miopen::Handle& /*profile_h*/,
                                         ConstData_t /*bot_ocl_buf*/,
                                         ConstData_t /*top_ocl_buf*/,
                                         Data_t /*wei_ocl_buf*/,
                                         ConstData_t /*bias_ocl_buf*/,
                                         const ConvolutionContext& /*params*/,
                                         const ConvSolution& /*solution*/,
                                         float& /*elapsed_time*/) const
    {
        MIOPEN_THROW("Not implemented");
    }
};

#define RUN_AND_MEASURE_HELPER_DECLARATION_FWD                     \
    int RunAndMeasureSolutionFwd(miopen::Handle& profile_h,        \
                                 ConstData_t bot_ocl_buf,          \
                                 Data_t top_ocl_buf,               \
                                 ConstData_t wei_ocl_buf,          \
                                 ConstData_t bias_ocl_buf,         \
                                 const ConvolutionContext& params, \
                                 const ConvSolution& solution,     \
                                 float& elapsed_time) const final;

#define RUN_AND_MEASURE_HELPER_DECLARATION_BWD                     \
    int RunAndMeasureSolutionBwd(miopen::Handle& profile_h,        \
                                 ConstData_t bot_ocl_buf,          \
                                 Data_t top_ocl_buf,               \
                                 ConstData_t wei_ocl_buf,          \
                                 ConstData_t bias_ocl_buf,         \
                                 const ConvolutionContext& params, \
                                 const ConvSolution& solution,     \
                                 float& elapsed_time) const final;

#define RUN_AND_MEASURE_HELPER_FROM_TEMPLATE_FWD(SOLVER)                   \
    int SOLVER::RunAndMeasureSolutionFwd(miopen::Handle& profile_h,        \
                                         ConstData_t bot_ocl_buf,          \
                                         Data_t top_ocl_buf,               \
                                         ConstData_t wei_ocl_buf,          \
                                         ConstData_t bias_ocl_buf,         \
                                         const ConvolutionContext& params, \
                                         const ConvSolution& solution,     \
                                         float& elapsed_time) const        \
    {                                                                      \
        return RunAndMeasureSolution(profile_h,                            \
                                     bot_ocl_buf,                          \
                                     top_ocl_buf,                          \
                                     wei_ocl_buf,                          \
                                     bias_ocl_buf,                         \
                                     params,                               \
                                     solution,                             \
                                     elapsed_time);                        \
    }

#define RUN_AND_MEASURE_HELPER_FROM_TEMPLATE_BWD(SOLVER)                   \
    int SOLVER::RunAndMeasureSolutionBwd(miopen::Handle& profile_h,        \
                                         Data_t bot_ocl_buf,               \
                                         ConstData_t top_ocl_buf,          \
                                         ConstData_t wei_ocl_buf,          \
                                         ConstData_t bias_ocl_buf,         \
                                         const ConvolutionContext& params, \
                                         const ConvSolution& solution,     \
                                         float& elapsed_time) const        \
    {                                                                      \
        return RunAndMeasureSolution(profile_h,                            \
                                     bot_ocl_buf,                          \
                                     top_ocl_buf,                          \
                                     wei_ocl_buf,                          \
                                     bias_ocl_buf,                         \
                                     params,                               \
                                     solution,                             \
                                     elapsed_time);                        \
    }

struct PerformanceConfigConvAsm3x3U final : Serializable<PerformanceConfigConvAsm3x3U>,
                                            IPerformanceConfig
{
    int limit_wave_cnt;        // [0..9]
    int filters_per_wave;      // [1..8]
    int output_lines_per_wave; // [1..8]

    PerformanceConfigConvAsm3x3U(int lwc, int fpw, int olpw);
    PerformanceConfigConvAsm3x3U() : PerformanceConfigConvAsm3x3U(-1, -1, -1) {}
    PerformanceConfigConvAsm3x3U(bool) : PerformanceConfigConvAsm3x3U(0, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.filters_per_wave, "filters_per_wave");
        f(self.output_lines_per_wave, "output_lines_per_wave");
    }

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue() override;
    bool IsValid(const ConvolutionContext& config) const final;
    bool operator==(const IPerformanceConfig& other) const final;
    std::string ToString() const;
};

struct ConvAsm3x3U final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const override { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const override;
    bool IsFast(const ConvolutionContext& params) const override;
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceConfigConvAsm3x3U>(sparce);
    }
    std::shared_ptr<IPerformanceConfig>
    GetPerformanceConfig(const ConvolutionContext&) const override;
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv) const final;

    RUN_AND_MEASURE_HELPER_DECLARATION_FWD
    RUN_AND_MEASURE_HELPER_DECLARATION_BWD

    private:
    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final;
};

struct PerformanceConfigConvAsm1x1U : Serializable<PerformanceConfigConvAsm1x1U>, IPerformanceConfig
{
    // ----------------- // Full set          Optimized       Spare
    // ----------------------------------------------------------------------------
    int read_size;        // [1..4]            <same>          <same>
    int k_mult;           // 1,[4,8,12..32]    2^n[8..32]      1,4
    int chunks_per_wave;  // [1..16]           [1..8]          <same>
    int chunk_size;       // 2^n[1..64]        2^n[16..64]     1,4
    int n_mult;           // [1..8]            [1..4]          <same>
    int c_mult;           // 2^n[1..32]        2^n[1..4]       <same>
    int waves_c_in_group; // [1..8]            [1..4]          <same>
    int waves_k_in_group; // 1,[2,4,8]         1,[2,4,8]       <same>
    bool use_spare_set;

    PerformanceConfigConvAsm1x1U(int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsm1x1U(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.read_size, "read_size");
        f(self.k_mult, "k_mult");
        f(self.chunks_per_wave, "chunks_per_wave");
        f(self.chunk_size, "chunk_size");
        f(self.n_mult, "n_mult");
        f(self.c_mult, "c_mult");
        f(self.waves_c_in_group, "waves_c_in_group");
        f(self.waves_k_in_group, "waves_k_in_group");
    }

    // clang-format off
    int GetReadSize() const { return read_size; }
    int GetKMult() const { return k_mult; }
    int GetChunksPerWave() const { return chunks_per_wave; }
    int GetChunkSize() const { return chunk_size; }
    int GetNMult() const { return n_mult; }
    int GetCMult() const { return c_mult; }
    int GetWavesCInGroup() const { return waves_c_in_group; }
    int GetWavesKInGroup() const { return waves_k_in_group; }
    int GetNPerGpr() const { assert(chunk_size); return 64 / chunk_size; }
    // clang-format on

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue() final;
    bool IsValid(const ConvolutionContext& config) const override;
    bool operator==(const IPerformanceConfig& other) const override;
    std::string ToString() const;
};

struct ConvAsm1x1UBase : virtual SearchableSolver<ConvolutionContext>
{
    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final;
    bool IsApplicable(const ConvolutionContext& params) const final;
    bool IsFast(const ConvolutionContext& params) const final;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv) const final;
};

struct ConvAsm1x1U final : ConvAsm1x1UBase, GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const override { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const ConvolutionContext&) const final;
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceConfigConvAsm1x1U>(sparce);
    }
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;

    RUN_AND_MEASURE_HELPER_DECLARATION_FWD
    RUN_AND_MEASURE_HELPER_DECLARATION_BWD
};

struct PerformanceConfigConvBiasActivAsm1x1U final : PerformanceConfigConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U(bool spare) : PerformanceConfigConvAsm1x1U(spare) {}
    PerformanceConfigConvBiasActivAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    bool IsValid(const ConvolutionContext& config) const final;
    bool operator==(const IPerformanceConfig& other) const final;
};

struct ConvBiasActivAsm1x1U final : ConvAsm1x1UBase, GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const override { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const ConvolutionContext&) const final;
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceConfigConvBiasActivAsm1x1U>(sparce);
    }
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;

    RUN_AND_MEASURE_HELPER_DECLARATION_FWD
    RUN_AND_MEASURE_HELPER_DECLARATION_BWD
};

struct PerformanceConfigConvAsm1x1UV2 final : Serializable<PerformanceConfigConvAsm1x1UV2>,
                                              IPerformanceConfig
{
    // ----------------- // Full set          Optimized       Spare
    // ----------------------------------------------------------------------------
    int chunk_size;       // 2^n[1..64]        2^n[16..64]     <same>
    int dwords_per_ld;    // [1..4]            1,2,3           <same>
    int k_mult;           // [1..32]           8,16            1,2,3,4
    int c_mult;           // [1..32]           2^n[1..4]       <same>
    int n_mult;           // [1..32]           1,2             <same>
    int w_mult;           // [1..32]           1,2             <same>
    int h_mult;           // [1..32]           1,2             <same>
    int h_per_chunk;      // 2^n[1..64]        [2,4,8]         <same>
    int waves_k_in_group; // [1..8]            2,4             <same>
    int waves_c_in_group; // [1..8]            1,2             <same>
    bool use_spare_set;

    PerformanceConfigConvAsm1x1UV2(int, int, int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1UV2()
        : PerformanceConfigConvAsm1x1UV2(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsm1x1UV2(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.chunk_size, "chunk_size");
        f(self.dwords_per_ld, "dwords_per_ld");
        f(self.k_mult, "k_mult");
        f(self.c_mult, "c_mult");
        f(self.n_mult, "n_mult");
        f(self.w_mult, "w_mult");
        f(self.h_mult, "h_mult");
        f(self.h_per_chunk, "h_per_chunk");
        f(self.waves_k_in_group, "waves_k_in_group");
        f(self.waves_c_in_group, "waves_c_in_group");
    }

    // clang-format off
    int GetChunkSize() const { return chunk_size; }
    int GetDwordsPerLd() const { return dwords_per_ld; }
    int GetCMult() const { return c_mult; }
    int GetKMult() const { return k_mult; }
    int GetNMult() const { return n_mult; }
    int GetWMult() const { return w_mult; }
    int GetHMult() const { return h_mult; }
    int GetHPerChunk() const { return h_per_chunk; }
    int GetWavesCInGroup() const { return waves_c_in_group; }
    int GetWavesKInGroup() const { return waves_k_in_group; }
    int GetNPerGpr() const { assert(chunk_size); return 64 / chunk_size; }
    // clang-format on

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue() final;
    bool IsValid(const ConvolutionContext& config) const final;
    bool operator==(const IPerformanceConfig& other) const final;
    std::string ToString() const;
};

struct ConvAsm1x1UV2 final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceConfigConvAsm1x1UV2>(sparce);
    }
    std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const ConvolutionContext&) const final;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final;
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    bool IsApplicable(const ConvolutionContext& params) const final;
    bool IsFast(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv) const final;

    RUN_AND_MEASURE_HELPER_DECLARATION_FWD
    RUN_AND_MEASURE_HELPER_DECLARATION_BWD
};

struct ConvAsm5x10u2v2f1 final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvAsm5x10u2v2b1 final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvAsm7x7c3h224w224k64u2v2p3q3f1 final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvOclDirectFwd11x11 final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvOclDirectFwdGen final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvOclDirectFwd3x3 final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct PerformanceImplicitGemm final : Serializable<PerformanceImplicitGemm>, IPerformanceConfig
{
    int BPerBlock; // 2^n[8..16]
    int KPerBlock; // 2^n[32..128]
    int EPerBlock; // 2^n[4..16]

    int GemmNRepeat; // == 2

    int GemmMPerThreadSubC; // 2^n[2..4]
    int GemmNPerThreadSubC; // 2^n[2..4]

    int GemmMLevel0Cluster; // 2^n[1..4]
    int GemmNLevel0Cluster; // 2^n[1..4]
    int GemmMLevel1Cluster; // 2^n[1..4]
    int GemmNLevel1Cluster; // 2^n[1..4]

    int InBlockCopyClusterLengths_E;  // 2^n[4..16]
    int InBlockCopyClusterLengths_B;  // 2^n[8..16]
    int InBlockCopyClusterLengths_N1; // 2^n[1..2]
    int InBlockCopyClusterLengths_N2; // 2^n[1..4]

    int WeiBlockCopyClusterLengths_E; // 2^n[1..4]
    int WeiBlockCopyClusterLengths_K; // 2^n[16..128]

    bool use_spare_set;

    PerformanceImplicitGemm(
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, bool);

    PerformanceImplicitGemm()
        : PerformanceImplicitGemm(
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemm(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BPerBlock, "BPerBlock");
        f(self.KPerBlock, "KPerBlock");
        f(self.EPerBlock, "EPerBlock");
        f(self.GemmNRepeat, "GemmNRepeat");
        f(self.GemmMPerThreadSubC, "GemmMPerThreadSubC");
        f(self.GemmNPerThreadSubC, "GemmNPerThreadSubC");
        f(self.GemmMLevel0Cluster, "GemmMLevel0Cluster");
        f(self.GemmNLevel0Cluster, "GemmNLevel0Cluster");
        f(self.GemmMLevel1Cluster, "GemmMLevel1Cluster");
        f(self.GemmNLevel1Cluster, "GemmNLevel1Cluster");
        f(self.InBlockCopyClusterLengths_E, "InBlockCopyClusterLengths_E");
        f(self.InBlockCopyClusterLengths_N1, "InBlockCopyClusterLengths_N1");
        f(self.InBlockCopyClusterLengths_B, "InBlockCopyClusterLengths_B");
        f(self.InBlockCopyClusterLengths_N2, "InBlockCopyClusterLengths_N2");
        f(self.WeiBlockCopyClusterLengths_E, "WeiBlockCopyClusterLengths_E");
        f(self.WeiBlockCopyClusterLengths_K, "WeiBlockCopyClusterLengths_K");
    }

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue() final;
    bool IsValid(const ConvolutionContext& ctx) const final;
    bool operator==(const IPerformanceConfig& other) const final;
    std::string ToString() const;
};

struct ConvHipImplicitGemmV4Fwd final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool) const final
    {
        return std::make_shared<PerformanceImplicitGemm>();
    }
    std::shared_ptr<IPerformanceConfig>
    GetPerformanceConfig(const ConvolutionContext& params) const final;
    bool IsValidPerformanceConfig(const ConvolutionContext& problem,
                                  const IPerformanceConfig& c) const final;
    bool IsApplicable(const ConvolutionContext& ctx) const final;

    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    int RunAndMeasureSolutionFwd(miopen::Handle& profile_h,
                                 ConstData_t bot_ocl_buf,
                                 Data_t top_ocl_buf,
                                 ConstData_t wei_ocl_buf,
                                 ConstData_t bias_ocl_buf,
                                 const ConvolutionContext& params,
                                 const ConvSolution& solution,
                                 float& elapsed_time) const final;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv) const final;
};

struct PerformanceImplicitGemmXdlops final : Serializable<PerformanceImplicitGemmXdlops>,
                                             IPerformanceConfig
{
    int BPerBlock; // 2^n[8..16]
    int KPerBlock; // 2^n[32..128]
    int EPerBlock; // 2^n[4..16]

    int GemmMPerWave;
    int GemmNPerWave;

    int InBlockCopyClusterLengths_E; // 2^n[4..16]
    int InBlockCopyClusterLengths_B; // 2^n[8..16]

    int WeiBlockCopyClusterLengths_E; // 2^n[1..4]
    int WeiBlockCopyClusterLengths_K; // 2^n[16..128]

    bool use_spare_set;

    PerformanceImplicitGemmXdlops(int, int, int, int, int, int, int, int, int, bool);

    PerformanceImplicitGemmXdlops()
        : PerformanceImplicitGemmXdlops(-1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmXdlops(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BPerBlock, "BPerBlock");
        f(self.KPerBlock, "KPerBlock");
        f(self.EPerBlock, "EPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.InBlockCopyClusterLengths_E, "InBlockCopyClusterLengths_E");
        f(self.InBlockCopyClusterLengths_B, "InBlockCopyClusterLengths_B");
        f(self.WeiBlockCopyClusterLengths_E, "WeiBlockCopyClusterLengths_E");
        f(self.WeiBlockCopyClusterLengths_K, "WeiBlockCopyClusterLengths_K");
    }

    void EuristicInit(const ConvolutionContext& ctx);
    bool IsValidValue() const;
    bool SetNextValue() final;
    bool IsValid(const ConvolutionContext& ctx) const final;
    bool operator==(const IPerformanceConfig& other) const final;
    std::string ToString() const;
};

struct ConvHipImplicitGemmV4R4FwdXdlops final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig>
    GetPerformanceConfig(const ConvolutionContext& ctx) const final;
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceImplicitGemmXdlops>(sparce);
    }
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const IPerformanceConfig& c) const final;
    bool IsApplicable(const ConvolutionContext& ctx) const final;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv) const final;

    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    int RunAndMeasureSolutionFwd(miopen::Handle& profile_h,
                                 ConstData_t bot_buf,
                                 Data_t top_buf,
                                 ConstData_t wei_buf,
                                 ConstData_t bias_buf,
                                 const ConvolutionContext& ctx,
                                 const ConvSolution& solution,
                                 float& elapsed_time) const final;
};

struct ConvHipImplicitGemmV4R4Xdlops_1x1 final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig>
    GetPerformanceConfig(const ConvolutionContext& ctx) const final;
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceImplicitGemmXdlops>(sparce);
    }
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const IPerformanceConfig& c) const final;
    bool IsApplicable(const ConvolutionContext& ctx) const final;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv = false) const final;

    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    int RunAndMeasureSolutionFwd(miopen::Handle& profile_h,
                                 ConstData_t bot_buf,
                                 Data_t top_buf,
                                 ConstData_t wei_buf,
                                 ConstData_t bias_buf,
                                 const ConvolutionContext& ctx,
                                 const ConvSolution& solution,
                                 float& elapsed_time) const final;
};

struct ConvHipImplicitGemmV4_1x1 final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig>
    GetPerformanceConfig(const ConvolutionContext& ctx) const final;
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceImplicitGemm>(sparce);
    }
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const IPerformanceConfig& c) const final;
    bool IsApplicable(const ConvolutionContext& ctx) const final;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv = false) const final;

    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    int RunAndMeasureSolutionFwd(miopen::Handle& profile_h,
                                 ConstData_t bot_buf,
                                 Data_t top_buf,
                                 ConstData_t wei_buf,
                                 ConstData_t bias_buf,
                                 const ConvolutionContext& ctx,
                                 const ConvSolution& solution,
                                 float& elapsed_time) const final;
};

struct ConvHipImplicitGemmV4WrW final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig>
    GetPerformanceConfig(const ConvolutionContext& ctx) const final;
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceImplicitGemm>(sparce);
    }
    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const IPerformanceConfig& c) const final;
    bool IsApplicable(const ConvolutionContext& ctx) const final;
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv = false) const final;

    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    int RunAndMeasureSolutionFwd(miopen::Handle& profile_h,
                                 ConstData_t bot_buf,
                                 Data_t top_buf,
                                 ConstData_t wei_buf,
                                 ConstData_t bias_buf,
                                 const ConvolutionContext& ctx,
                                 const ConvSolution& solution,
                                 float& elapsed_time) const final;
};

/// Holds common member functions for the Solvers which share the same
/// "legacy exhaustive search" machinery.
struct ConvOclDirectFwdLegacyExhaustiveSearch : SearchableSolver<ConvolutionContext>
{
    std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const ConvolutionContext&) const final;
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;

    private:
    template <typename Tgpu>
    LegacyPerformanceConfig SearchImpl(const ConvolutionContext&) const;
};

struct ConvOclDirectFwdBase : ConvOclDirectFwdLegacyExhaustiveSearch
{
    bool IsApplicable(const ConvolutionContext& params) const final;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final;

    protected:
    bool IsApplicableBase(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwd final : ConvOclDirectFwdBase
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& searched_params,
                             bool disableConfigOverrideFromEnv) const final;
};

struct ConvOclDirectFwdFused final : ConvOclDirectFwdBase
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& searched_params,
                             bool disableConfigOverrideFromEnv) const final;
};

struct ConvOclDirectFwd1x1 final : ConvOclDirectFwdLegacyExhaustiveSearch
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final
    {
        return true;
    }
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& searched_params,
                             bool disableConfigOverrideFromEnv) const final;
};

struct ConvBinWinograd3x3U final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvBinWinogradRxS final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvBinWinogradRxSf3x2 final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvBinWinogradRxSFused final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvWinograd3x3MultipassWrW final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;

    // kernel_file_name for solver identification
    static std::string GetSolverFileNames(int id)
    {
        static const std::string names[3] = {"xform_data.s", "xform_filter.s", "xform_out.s"};
        return names[id];
    }
    static std::string GetSolverKernelNames(int id)
    {
        static const std::string name_suffix =
            '_' + std::to_string(WinoDataH) + '_' + std::to_string(WinoDataW) + '_' +
            std::to_string(WinoFilterH) + '_' + std::to_string(WinoFilterW);
        static const std::string names[3] = {"gcnAsmWinogradXformData" + name_suffix,
                                             "gcnAsmWinogradXformFilter" + name_suffix,
                                             "gcnAsmWinogradXformOut" + name_suffix};

        return names[id];
    }
    static int GetGroupCountMult() { return 4; }

    static int GetSolverWinoXformHWSize(const miopen::ConvolutionContext& ctx, int id)
    {
        if(id == 0)
            return WinoDataH + (WinoFilterH - 1) * (WinoDataH == 7 ? 2 : ctx.kernel_stride_h);
        else
            return WinoDataW + (WinoFilterW - 1) * (WinoDataW == 7 ? 2 : ctx.kernel_stride_w);
    }
};

extern template struct ConvWinograd3x3MultipassWrW<3, 2>;
extern template struct ConvWinograd3x3MultipassWrW<3, 3>;
extern template struct ConvWinograd3x3MultipassWrW<3, 4>;
extern template struct ConvWinograd3x3MultipassWrW<3, 5>;
extern template struct ConvWinograd3x3MultipassWrW<3, 6>;
extern template struct ConvWinograd3x3MultipassWrW<7, 2>;
extern template struct ConvWinograd3x3MultipassWrW<7, 3>;
extern template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 2>;
extern template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 3>;
extern template struct ConvWinograd3x3MultipassWrW<7, 2, 1, 1>;
extern template struct ConvWinograd3x3MultipassWrW<7, 3, 1, 1>;

struct PerformanceConfigAsmDirect3x3WrW final : Serializable<PerformanceConfigAsmDirect3x3WrW>,
                                                IPerformanceConfig
{
    int limit_wave_cnt;   // [0..9]
    int reverse_inout;    // [0..1], 1 is allowed for stride=1x1 only.
    int chunk_size;       // {16,8}, Smaller values increase register pressure.
    int k_per_wave;       // {1,2,4,8} && ((chunk_size * k_per_wave) <= 64).
                          // Higher values increase register pressure.
    int pipe_lines_depth; // [1..16] && (pipe_lines_depth <= img_h).
                          // Higher values increase register pressure.
    int n_per_group;      // [1..8] && (n_per_group <= batch_size).

    PerformanceConfigAsmDirect3x3WrW(int lwc, int rio, int csz, int kpw, int pld, int npg);
    PerformanceConfigAsmDirect3x3WrW() : PerformanceConfigAsmDirect3x3WrW(-1, -1, -1, -1, -1, -1) {}
    PerformanceConfigAsmDirect3x3WrW(bool) : PerformanceConfigAsmDirect3x3WrW(0, 0, 8, 1, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.reverse_inout, "reverse_inout");
        f(self.chunk_size, "chunk_size");
        f(self.k_per_wave, "k_per_wave");
        f(self.pipe_lines_depth, "pipe_lines_depth");
        f(self.n_per_group, "n_per_group");
    }

    // clang-format off
    int GetLimitWaveCnt() const { return limit_wave_cnt; }
    int GetReverseInout() const { return reverse_inout; }
    int GetChunkSize() const { return chunk_size; }
    int GetKPerWave() const { return k_per_wave; }
    int GetPipeLinesDepth() const { return pipe_lines_depth; }
    int GetNPerGroup() const { return n_per_group; }
    int GetCPerWave() const { assert(chunk_size); return 64 / chunk_size; } // clang-format on

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue() final;
    bool IsValid(const ConvolutionContext& config) const final;
    bool operator==(const IPerformanceConfig& other) const final;
    std::string ToString() const;
};

struct ConvAsmBwdWrW3x3 final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceConfigAsmDirect3x3WrW>(sparce);
    }
    std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const ConvolutionContext&) const final;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final;
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    bool IsApplicable(const ConvolutionContext& params) const final;
    bool IsFast(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv) const final;

    RUN_AND_MEASURE_HELPER_DECLARATION_FWD
    RUN_AND_MEASURE_HELPER_DECLARATION_BWD
};

struct PerformanceConfigConvAsmBwdWrW1x1 final : Serializable<PerformanceConfigConvAsmBwdWrW1x1>,
                                                 IPerformanceConfig
{
    int chunk_size;    // {1,2,4,8,16}
    int c_per_gpr;     // {1,2,4,8,16}
    int c_mult;        // {1,2,4,8,16}
    int k_per_gpr;     // {1,2,4,8,16}
    int k_mult;        // {1,2,4,8,16}
    int n_per_gpr;     // {1,2,4}
    int n_part_cnt;    // [1..8]
    int read_size;     // [1..4]
    int short_store;   // {0,1}
    int data_prefetch; // [0..4]
    bool use_spare_set;

    /// The following conditions must be met.
    ///
    /// Shader design-related constraints:
    /// - (A) (chunk_size * c_per_gpr) == 16
    /// - (B) k_per_gpr <= c_per_gpr
    /// - (C) (c_mult > 1 || k_mult > 1)
    ///         ? ((fwd_C % (c_per_gpr * c_mult) == 0) && (fwd_K % (k_per_gpr * k_mult) == 0))
    ///         : (true)
    ///
    /// Resource-related constraints:
    /// - (D) c_mult * k_mult * k_per_gpr + 9 + (c_mult + k_mult) * read_size * pipe_depth <= 256
    ///
    /// Where:
    /// - fwd_C := Num input channels for forward convolution (-c).
    ///   For backward, this is actually n_outputs.
    /// - fwd_K := Num output channels for forward convolution (-k).
    ///   For backward, this is actually n_inputs.

    PerformanceConfigConvAsmBwdWrW1x1(int chunk_size_,
                                      int c_per_gpr_,
                                      int c_mult_,
                                      int k_per_gpr_,
                                      int k_mult_,
                                      int n_per_gpr_,
                                      int n_part_cnt_,
                                      int read_size_,
                                      int short_store_,
                                      int data_prefetch_,
                                      bool);
    PerformanceConfigConvAsmBwdWrW1x1()
        : PerformanceConfigConvAsmBwdWrW1x1(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsmBwdWrW1x1(bool spare)
        : PerformanceConfigConvAsmBwdWrW1x1(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.chunk_size, "chunk_size");
        f(self.c_per_gpr, "c_per_gpr");
        f(self.c_mult, "c_mult");
        f(self.k_per_gpr, "k_per_gpr");
        f(self.k_mult, "k_mult");
        f(self.n_per_gpr, "n_per_gpr");
        f(self.n_part_cnt, "n_part_cnt");
        f(self.read_size, "read_size");
        f(self.short_store, "short_store");
        f(self.data_prefetch, "data_prefetch");
    }

    // clang-format off
    int GetChunkSize() const { return chunk_size; }
    int GetCPerGpr() const { return c_per_gpr; }
    int GetCMult() const { return c_mult; }
    int GetKPerGpr() const { return k_per_gpr; }
    int GetKMult() const { return k_mult; }
    int GetNPerGpr() const { return n_per_gpr; }
    int GetNPartCnt() const { return n_part_cnt; }
    int GetHWPerGpr() const {   assert(c_per_gpr); assert(n_per_gpr); assert(chunk_size);
                                return wave_size / (c_per_gpr * n_per_gpr * chunk_size); } // "hw" stands for "height-and-width".
    int GetReadSize() const { return read_size; }
    int GetShortStore() const {return short_store; }
    int GetDataPrefetch() const { return data_prefetch; }
    // clang-format on

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue() final;
    bool IsValid(const ConvolutionContext& config) const final;
    bool operator==(const IPerformanceConfig& other) const final;
    std::string ToString() const;
};

struct ConvAsmBwdWrW1x1 final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceConfigConvAsmBwdWrW1x1>(sparce);
    }
    std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const ConvolutionContext&) const final;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const final;
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    bool IsApplicable(const ConvolutionContext& params) const final;
    bool IsFast(const ConvolutionContext& params) const final;
    int RunAndMeasureSolutionWrW(miopen::Handle& profile_h,
                                 ConstData_t bot_ocl_buf,
                                 ConstData_t top_ocl_buf,
                                 Data_t wei_ocl_buf,
                                 ConstData_t bias_ocl_buf,
                                 const ConvolutionContext& params,
                                 const ConvSolution& solution,
                                 float& elapsed_time) const final;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv) const final;
};

/// N_BATCH_LOOPS - {1,2,4,8,16} Num batches processed in single workitem.
///     Required workspace size depends on it. However there is a restriction in the internal
///     Solver API that this shouldn't be so. Therefore the family of Solvers created.
///     Each Solver in the family has constant value of this parameter.
template <int N_BATCH_LOOPS>
struct PerformanceConfigConvOclBwdWrw2 final
    : Serializable<PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>>,
      IPerformanceConfig
{
    // Num waves involved a workgroup.
    int n_waves = -1; // {1,2,4,8}
    // Num values to read in a workitem (read_unit).
    int read_size = -1; // [6..12]
    // Num of output channels (top/bottom layer in forward/backward direction)
    // that share the same input channel in single workgroup.
    // Also represents number of output channels in single tile.
    int n_out_channels_per_tile = -1; // {1,2,4,8}
    // How many tiles of output channels are processed in a single workgroup?
    // n_out_channels_in_lcl * n_out_channels_tiles = total number of
    // output channels processed in single workgroup.
    int n_out_channels_tiles = -1; // {1,2,4,8}
    // Num of output rows processed in a single iteration of loop in a workitem
    // (N_ALIGNED_OUT_SCAN_BLK).
    int n_out_rows_in_lcl = -1; // [2..11]

    PerformanceConfigConvOclBwdWrw2(int nw, int rs, int nocpt, int noct, int noril)
        : n_waves(nw),
          read_size(rs),
          n_out_channels_per_tile(nocpt),
          n_out_channels_tiles(noct),
          n_out_rows_in_lcl(noril)
    {
    }
    PerformanceConfigConvOclBwdWrw2() {}
    PerformanceConfigConvOclBwdWrw2(bool) : PerformanceConfigConvOclBwdWrw2(1, 6, 1, 1, 2) {}
    // spare_set is not used in this solver.

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.n_waves, "n_waves");
        f(self.read_size, "read_size");
        f(self.n_out_channels_per_tile, "n_out_channels_per_tile");
        f(self.n_out_channels_tiles, "n_out_channels_tiles");
        f(self.n_out_rows_in_lcl, "n_out_rows_in_lcl");
    }

    // clang-format off
    int GetNumWaves() const { return n_waves; }
    int GetReadSize() const { return read_size; }
    int GetNumOutChannelsPerTile() const { return n_out_channels_per_tile; }
    int GetNumOutChannelTiles() const { return n_out_channels_tiles; }
    int GetNumOutRowsPerIterPerWork() const { return n_out_rows_in_lcl; } // clang-format on

    void EuristicInit(const ConvolutionContext& params);
    bool IsValidValue() const;
    bool SetNextValue() final;
    bool IsValid(const ConvolutionContext& params) const final;
    bool operator==(const IPerformanceConfig& other) const final;
    std::string ToString() const;
};

template <int N_BATCH_LOOPS>
struct ConvOclBwdWrW2Base : virtual SolverBase<ConvolutionContext>
{
    protected:
    bool IsApplicableBase(const ConvolutionContext& params) const;
    ConvSolution GetSolutionBase(const ConvolutionContext& params,
                                 const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& config,
                                 bool disableConfigOverrideFromEnv = false) const;
    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
    GetPerformanceConfigBase(const ConvolutionContext&) const;
    bool IsValidPerformanceConfigBase(const ConvolutionContext&,
                                      const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&) const;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const final;
};

template <int N_BATCH_LOOPS>
struct ConvOclBwdWrW2 final : GenericSearchableSolver<ConvolutionContext>,
                              ConvOclBwdWrW2Base<N_BATCH_LOOPS>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>>(sparce);
    }
    bool IsApplicable(const ConvolutionContext& params) const final;
    std::shared_ptr<IPerformanceConfig>
    GetPerformanceConfig(const ConvolutionContext& context) const final
    {
        return std::make_shared<PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>>(
            this->GetPerformanceConfigBase(context));
    }
    bool IsValidPerformanceConfig(const ConvolutionContext& context,
                                  const IPerformanceConfig& config_) const final
    {
        const auto& config =
            dynamic_cast<const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&>(config_);
        return this->IsValidPerformanceConfigBase(context, config);
    }
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    int RunAndMeasureSolutionWrW(miopen::Handle& profile_h,
                                 ConstData_t bot_ocl_buf,
                                 ConstData_t top_ocl_buf,
                                 Data_t wei_ocl_buf,
                                 ConstData_t bias_ocl_buf,
                                 const ConvolutionContext& context,
                                 const ConvSolution& solution,
                                 float& elapsed_time) const final;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& config_,
                             bool disableConfigOverrideFromEnv) const final
    {
        const auto& config =
            dynamic_cast<const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&>(config_);
        return this->GetSolutionBase(params, config, disableConfigOverrideFromEnv);
    }

    private:
    template <typename Tgpu>
    int RunAndMeasureSolutionImpl(miopen::Handle& profile_h,
                                  ConstData_t bot_ocl_buf,
                                  ConstData_t top_ocl_buf,
                                  Data_t wei_ocl_buf,
                                  ConstData_t bias_ocl_buf,
                                  const ConvolutionContext& context,
                                  const ConvSolution& solution,
                                  float& elapsed_time) const;
};

// To suppress misleading warning
#ifndef CONV_OCL_DIR2D_BWDWRW_2_CPP
extern template struct ConvOclBwdWrW2<1>;
extern template struct ConvOclBwdWrW2<2>;
extern template struct ConvOclBwdWrW2<4>;
extern template struct ConvOclBwdWrW2<8>;
extern template struct ConvOclBwdWrW2<16>;
#endif

/// A separate solver from ConvOclBwdWrW2 to disable auto-tuning for certain configs.
/// Basically, this is *hack* for non-group 3x3 and 1x1 cases.
/// It is assumed that Solutions provided by the ConvOclBwdWrW2 solver
/// would never beat 3x3 and 1x1 assembly WrW kernels, even after tuning.
struct ConvOclBwdWrW2NonTunable final : ConvOclBwdWrW2Base<1>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
};

struct ConvOclBwdWrW53 final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const final;
};

struct ConvOclBwdWrW1x1 final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params) const final;
    size_t GetWorkspaceSize(const ConvolutionContext& params) const final;
};

#if MIOPEN_USE_SCGEMM
template <SCGemmOpType T>
struct PerformanceConfigSCGemmFwd final : Serializable<PerformanceConfigSCGemmFwd<T>>,
                                          IPerformanceConfig
{
    int routine = -1; //[0..6]

    PerformanceConfigSCGemmFwd();
    PerformanceConfigSCGemmFwd(bool);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.routine, "routine");
    }

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue() final;
    bool IsValid(const ConvolutionContext& config) const final;
    bool operator==(const IPerformanceConfig& other) const final;
    std::string ToString() const;
};
struct ConvSCGemmFwd final : GenericSearchableSolver<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    std::shared_ptr<IPerformanceConfig> GetGenericSearchStart(bool sparce) const final
    {
        return std::make_shared<PerformanceConfigSCGemmFwd<SCGemmOpType>>(sparce);
    }
    std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const ConvolutionContext&) const final;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final;
    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final;
    bool IsApplicable(const ConvolutionContext& params) const final;
    bool IsFast(const ConvolutionContext& params) const final;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const IPerformanceConfig& config,
                             bool disableConfigOverrideFromEnv) const final;

    RUN_AND_MEASURE_HELPER_DECLARATION_FWD
    RUN_AND_MEASURE_HELPER_DECLARATION_BWD

    protected:
    bool IsApplicableBase(const ConvolutionContext& params) const;
};

extern template struct PerformanceConfigSCGemmFwd<SCGemmOpFGemm>;
#endif

/// Partial implementation.
struct gemm final : SolverBase<ConvolutionContext>
{
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& /*params*/) const final { return false; };
    ConvSolution GetSolution(const ConvolutionContext&) const final
    {
        return ConvSolution{miopenStatusNotInitialized};
    }
};

} // namespace solver

using ConvSolver = solver::SolverBase<ConvolutionContext>;

} // namespace miopen

struct mlo_construct_direct2D_fusion : mlo_construct_base
{
    mlo_construct_direct2D_fusion(int dir, bool do_bias = false) : mlo_construct_base(dir, do_bias)
    {
    }
    mlo_construct_direct2D_fusion(const miopen::TensorDescriptor& in,
                                  const miopen::TensorDescriptor& weights,
                                  const miopen::TensorDescriptor& out,
                                  const miopen::ConvolutionDescriptor& conv,
                                  int dir,
                                  bool do_bias = false)
        : mlo_construct_base(in, weights, out, conv, dir, do_bias)
    {
    }

    inline void mloCopyTo(miopen::ConvolutionContext& params) const /// TODO: get rid of this
    {
        params = _search_params;
    }
    miopen::solver::ConvSolution FindSolution(const std::vector<miopen::ConvSolver*>& solvers);
};

#endif // GUARD_MIOPEN_SOLVER_HPP_
