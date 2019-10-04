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

#ifndef GUARD_MIOPEN_GENERIC_SEARCH_HPP_
#define GUARD_MIOPEN_GENERIC_SEARCH_HPP_

#include <miopen/config.h>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/solver.hpp>

#include <chrono>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <vector>

namespace miopen {
namespace solver {

/// This STL-like container together with corresponding iterator provide access
/// to a set of all available performance configs for the given problem config.
///
/// Implementation does not hold values themselves as these would take too much memory.
/// The container holds problem config information instead. This info
/// is required for advancing the iterator to the next valid configuration.
template <typename Context>
class ComputedContainer;

template <typename Context>
class ComputedIterator
    : public std::iterator<std::input_iterator_tag, std::shared_ptr<IPerformanceConfig>>
{
    std::shared_ptr<IPerformanceConfig> v;
    const Context* p; // For Next().

    ComputedIterator& Next()
    {
        if(p != nullptr)
        {
            do
            {
                if(!v->SetNextValue())
                { // Wraparound, end reached. Iterator is useless from now.
                    p = nullptr;
                    break;
                }
            } while(!v->IsValid(*p));
        }
        return *this;
    }

    // Implements container's begin()
    ComputedIterator(const Context& problem,
                     const bool spare,
                     const GenericSearchableSolver<Context>& solver)
        : v(solver.GetGenericSearchStart(spare)), p(&problem)
    {
        if(!v->IsValid(*p))
            Next();
    }

    public:
    // STL-like iterator shall be default contructible. Also implements container's end()
    ComputedIterator() : v(), p(nullptr) {}
    // STL-like iterator shall be copy contructible. The default copy ctor is ok.

    ComputedIterator& operator++() { return Next(); }
    const std::shared_ptr<IPerformanceConfig>& operator*() const { return v; }
    bool operator!=(ComputedIterator const& other) const
    {
        if(p == other.p)
            if(p == nullptr // Ends are always equal.
               ||
               v == other.v)
                return false;
        return true;
    }
    bool operator==(ComputedIterator const& other) const { return !(*this != other); }

    friend class ComputedContainer<Context>;
};

template <typename Context>
class ComputedContainer
{
    using Solver = GenericSearchableSolver<Context>;

    Context problem; // Hold a copy make the object independent of the environment.
    bool spare;      // Use spare set of perf configs. Those are usually slower than main set.
                     // Splitting the theoretically available set of perf configs to "main"
                     // and "spare" sets allows for acceleration of the auto-tune process:
                     // * If the "main" set is not empty, then skipping the "spare" set
                     //   avoids wasting time, because the latter is slower by definition.
                     // * Combining "spare" and "main" would lead to exponential growth of
                     //   the resulting container, and thus to exponential slowdown.
                     //
                     // Nevertheless, a Solver is free to either use or not use this capability
                     // (i.e. it is ok for PerformanceConfigInstance(bool) to ignore its parameter).
    const Solver& solver;

    /// \note We do not add 'const' to keep the object assignable
    /// for the sake of flexibility. Nevertheless, all element accesses of
    /// the "computed container" shall be const.

    public:
    using const_iterator = ComputedIterator<Context>;

    ComputedContainer(const Context& problem_, const Solver& solver_, const bool spare_ = false)
        : problem(problem_), spare(spare_), solver(solver_)
    {
    }
    const_iterator begin() const { return {problem, spare, solver}; }
    const_iterator end() const { return {}; }
};

class Timer
{
    public:
    Timer(){};
    void start() { st = std::chrono::steady_clock::now(); }
    float elapsed_ms()
    {
        capture();
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et - st)
            .count();
    }

    private:
    void capture() { et = std::chrono::steady_clock::now(); }
    std::chrono::time_point<std::chrono::steady_clock> st;
    std::chrono::time_point<std::chrono::steady_clock> et;
};

class HeartBeat
{
    size_t n_within_beat;
    size_t n_best;
    float best_time; // within beat
    float elapsed_cumulative;
    Timer timer;
    std::shared_ptr<IPerformanceConfig> best_config;

    void Continue()
    {
        best_time     = std::numeric_limits<float>::max();
        n_within_beat = 0;
        timer.start();
    }

    public:
    HeartBeat() : n_within_beat(), n_best(), best_time(), elapsed_cumulative() {}

    void Start(const std::shared_ptr<IPerformanceConfig>& config)
    {
        elapsed_cumulative = 0.0f;
        best_config        = config;
        Continue();
    }

    void Monitor(const bool is_recent_failed,
                 const float recent_time,
                 const size_t n_recent,
                 const float total_best,
                 size_t n_failed,
                 size_t n_total,
                 const std::shared_ptr<IPerformanceConfig>& recent_config)
    {
        ++n_within_beat;
        if(!is_recent_failed && (recent_time < best_time))
        {
            best_time   = recent_time;
            n_best      = n_recent;
            best_config = recent_config;
        }
        const float elapsed = timer.elapsed_ms();
        if(elapsed > 3000)
        {
            elapsed_cumulative += elapsed;
            const float eta_sec =
                n_recent != 0u ? ((n_total - n_recent) * (elapsed_cumulative / n_recent) / 1000)
                               : 0.0f; // paraniod
            MIOPEN_LOG_W(n_recent << '/' << n_failed << '/' << n_total << ' ' << total_best
                                  << ", best within recent "
                                  << n_within_beat
                                  << ": "
                                  << best_time
                                  << " #"
                                  << n_best
                                  << ' '
                                  << best_config
                                  << ", ETA:"
                                  << eta_sec
                                  << " sec.");
            Continue();
        }
    }
};

inline void InitRandomly(std::vector<float>& vec, const double offset, const double factor)
{
    float* p = vec.data();
    for(unsigned long i = 0; i < vec.size(); ++i)
        *p++ = static_cast<float>((rand() * (1.0 / RAND_MAX) + offset) * factor);
}

inline void InitRandomly(std::vector<float>& vec)
{
    float* p = vec.data();
    for(unsigned long i = 0; i < vec.size(); ++i)
        *p++ = static_cast<float>(rand() * (1.0 / RAND_MAX));
}

inline size_t divide_round_plus_inf(const size_t x, const unsigned y)
{
    assert(/*x >= 0 &&*/ y > 0);
    if(x % y != 0)
        return x / y + 1;
    return x / y;
}

enum class Direction
{
    Forward,
    Bacward,
    Weights,
};

enum class SearchTweak
{
    None,
    /// Enforces the generic search algorithm
    /// to use workspace buffer instead of input data (x/dx) buffer.
    /// Example use case: Solution uses (non-tunable) subsampling or upsampling
    /// kernel which reads x/dx buffer and writes workspace, and then tunable
    /// convolution kernel which reads workspace instead of x/dx buffer.
    /// Another example: the first tunable kernel writes workspace (instead of x/dx),
    /// and the second non-tunable kernel converts workspace to user's buffer.
    WorkspaceInsteadOfXBuffer,
    /// This tweak is like previous one and use cases are similar,
    /// but it enforces the generic search algorithm
    /// to use workspace buffer instead of weights buffer.
    WorkspaceInsteadOfWeightsBuffer,
};

template <Direction direction>
struct RunAndMeasure
{
};

template <>
struct RunAndMeasure<Direction::Forward>
{
    template <class Context, class... Args>
    int operator()(const GenericSearchableSolver<Context>& s, Args&&... args) const
    {
        return s.RunAndMeasureSolutionFwd(args...);
    }
};

template <>
struct RunAndMeasure<Direction::Bacward>
{
    template <class Context, class... Args>
    int operator()(const GenericSearchableSolver<Context>& s, Args&&... args) const
    {
        return s.RunAndMeasureSolutionBwd(args...);
    }
};

template <>
struct RunAndMeasure<Direction::Weights>
{
    template <class Context, class... Args>
    int operator()(const GenericSearchableSolver<Context>& s, Args&&... args) const
    {
        return s.RunAndMeasureSolutionWrW(args...);
    }
};

template <Direction direction, class Context, typename TopT, typename BotT, typename WeiT>
std::shared_ptr<IPerformanceConfig> GenericSearch(const GenericSearchableSolver<Context>& s,
                                                  const Context& context,
                                                  const SearchTweak tweak,
                                                  TopT top_ocl_ptr,
                                                  BotT bot_ocl_ptr,
                                                  WeiT wei_ocl_ptr)
{
    std::shared_ptr<IPerformanceConfig> best_config;
    const auto default_config   = s.GetPerformanceConfig(context);
    const auto default_solution = s.GetSolution(context, *default_config);

    auto& profile_h          = context.GetStream();
    ConstData_t bias_ocl_ptr = context.GetBufs().bias;
    if(context.bias != 0 && bias_ocl_ptr == nullptr)
        MIOPEN_THROW("GenericSearch: context.bias != 0 && bias_ocl_ptr == nullptr");
    if(top_ocl_ptr == nullptr || bot_ocl_ptr == nullptr || wei_ocl_ptr == nullptr)
        MIOPEN_THROW("GenericSearch: top_ocl_ptr == nullptr || bot_ocl_ptr == nullptr || "
                     "wei_ocl_ptr == nullptr");
    switch(tweak)
    {
    case SearchTweak::None: break;
    case SearchTweak::WorkspaceInsteadOfXBuffer:
    {
        if(context.GetBufs().workSpaceSize < default_solution.workspce_sz ||
           context.GetBufs().workSpace == nullptr)
            MIOPEN_THROW("GenericSearch: Too small workspace or nullptr");
        if(context.direction.IsForward())
            bot_ocl_ptr = context.GetBufs().workSpace;
        else // bwd or wrw
            top_ocl_ptr = context.GetBufs().workSpace;
    }
    break;
    case SearchTweak::WorkspaceInsteadOfWeightsBuffer:
    {
        if(context.GetBufs().workSpaceSize < default_solution.workspce_sz ||
           context.GetBufs().workSpace == nullptr)
            MIOPEN_THROW("GenericSearch: Too small workspace or nullptr");
        wei_ocl_ptr = context.GetBufs().workSpace;
    }
    break;
    default: MIOPEN_THROW("GenericSearch: Unsupported SearchTweak value.");
    }

    AutoEnableProfiling enableProfiling{profile_h};

    using ConfigContainer = ComputedContainer<Context>;
    const ConfigContainer main(context, s);
    const int main_size = std::distance(main.begin(), main.end());
    const ConfigContainer spare(context, s, true);
    const int spare_size = std::distance(spare.begin(), spare.end());
    const bool useSpare  = (main_size == 0);

    const ConfigContainer& all_configs = useSpare ? spare : main;
    const int n_runs_total             = useSpare ? spare_size : main_size;
    MIOPEN_LOG_W(s.DbId() << ": Searching the best solution among " << n_runs_total
                          << (useSpare ? " (spare)" : "")
                          << "...");

    bool is_passed   = false; // left false only if all iterations failed.
    float best_time  = std::numeric_limits<float>::max();
    size_t n_failed  = 0;
    size_t n_current = 0;
    size_t n_best    = 0;
    HeartBeat heartbeat;
    heartbeat.Start(default_config);

    profile_h.EnableProfiling(true);
    for(auto& current_config_ptr : all_configs)
    {
        const auto& current_config = *current_config_ptr;
        float elapsed_time         = 0.0f;
        int ret                    = 0;
        MIOPEN_LOG_I2('#' << n_current << '/' << n_failed << '/' << n_runs_total << ' '
                          << current_config);

        const auto current_solution = s.GetSolution(context, current_config, true);
        if((tweak == SearchTweak::WorkspaceInsteadOfXBuffer ||
            tweak == SearchTweak::WorkspaceInsteadOfWeightsBuffer) &&
           default_solution.workspce_sz != current_solution.workspce_sz)
        {
            ret = -2;
            MIOPEN_LOG_E('#' << n_current << " (" << n_runs_total << ") "
                             << "Workspace size should not depend on PerformanceConfig: "
                             << default_solution.workspce_sz
                             << " != "
                             << current_solution.workspce_sz);
        }

        if(ret == 0)
        {
            ret = RunAndMeasure<direction>{}(s,
                                             profile_h,
                                             bot_ocl_ptr,
                                             top_ocl_ptr,
                                             wei_ocl_ptr,
                                             bias_ocl_ptr,
                                             context,
                                             current_solution,
                                             elapsed_time);
        }
        MIOPEN_LOG_T("##"
                     << "(n_current, n_failed, n_runs_total):  "
                     << n_current
                     << '/'
                     << n_failed
                     << '/'
                     << n_runs_total
                     << " elapsed_time: "
                     << elapsed_time
                     << ", best_time: "
                     << best_time
                     << ", "
                     << current_config);

        if(ret == 0)
        {
            // Smooth the jitter of measurements:
            // If the 1st probe is NOT too bad (measured time <= 1.05 * best known time),
            // then re-run it 4 times more and compute average time,
            // and decide using average of all 5 attempts vs. the best.
            if(elapsed_time / best_time < 1.05f)
            {
                MIOPEN_LOG_I2("Finding average for: " << elapsed_time << " / " << best_time << " = "
                                                      << (elapsed_time / best_time));
                float temp;
                for(int i = 0; i < 4; ++i)
                {
                    ret = RunAndMeasure<direction>{}(s,
                                                     profile_h,
                                                     bot_ocl_ptr,
                                                     top_ocl_ptr,
                                                     wei_ocl_ptr,
                                                     bias_ocl_ptr,
                                                     context,
                                                     current_solution,
                                                     temp);
                    if(ret != 0)
                    {
                        break;
                    }
                    elapsed_time += temp;
                }
                if(ret == 0)
                {
                    is_passed = true;
                    elapsed_time /= 5;
                    if(elapsed_time < best_time)
                    {
                        MIOPEN_LOG_I('#' << n_current << '/' << n_failed << '/' << n_runs_total
                                         << ' '
                                         << elapsed_time
                                         << " < "
                                         << best_time
                                         << ' '
                                         << current_config);
                        best_config = current_config_ptr;
                        best_time   = elapsed_time;
                        n_best      = n_current;
                    }
                    else
                    {
                        MIOPEN_LOG_I2(
                            "Average is not better: " << elapsed_time << " >= " << best_time);
                    }
                }
            }
        }

        if(ret != 0)
        {
            MIOPEN_LOG_E('#' << n_current << " (" << n_runs_total << ") "
                             << " Failed rc="
                             << ret);
            ++n_failed;
        }
        heartbeat.Monitor(ret != 0,
                          elapsed_time,
                          n_current,
                          best_time,
                          n_failed,
                          n_runs_total,
                          current_config_ptr);
        ++n_current;
    }

    profile_h.EnableProfiling(false);
    MIOPEN_LOG_W("Done: " << n_runs_total << '/' << n_failed << '/' << n_runs_total << ", best #"
                          << n_best
                          << ' '
                          << best_time
                          << ' '
                          << best_config);
    if(!is_passed)
        MIOPEN_THROW("Search failed");
    // Run once with the default config and show score.
    float default_time = 0.0f;
    profile_h.EnableProfiling(true);
    if(RunAndMeasure<direction>{}(s,
                                  profile_h,
                                  bot_ocl_ptr,
                                  top_ocl_ptr,
                                  wei_ocl_ptr,
                                  bias_ocl_ptr,
                                  context,
                                  default_solution,
                                  default_time) == 0)
    {
        const float score = (best_time > 0.0f) ? default_time / best_time : 0.0f;
        MIOPEN_LOG_W("...Score: " << score << " (default time " << default_time << ')');
    }
    profile_h.EnableProfiling(false);
    return best_config;
}

/// Solver member function requirements:
/// * GetPerformanceConfig shall be implemented.
///   - Its return type shall be suitable for instantiation of the ComputedContainer.
/// * GetSolution shall be implemented.
/// * RunAndMeasureSolution shall be implemented.
///
/// clang-format-off
/// -----------------------------------------------
/// Dataflow:
///      Forward:
///          wei[] (w) --> +--------+
///                        | kernel | --> top[] (y)
///          bot[] (x) --> +--------+
///
///      Backward data:
///          wei[] (w) --> +--------+
///                        | kernel | --> top[] (dx)
///         bot[] (dy) --> +--------+
///
///      Backward WrW:
///         top[] (dx) --> +--------+
///                        | kernel | --> wei[] (dw)
///         bot[] (dy) --> +--------+
/// ------------------------------------------------
/// clang-format-on
template <class Context>
std::shared_ptr<IPerformanceConfig> GenericSearchFwd(const GenericSearchableSolver<Context>& s,
                                                     const Context& context,
                                                     const SearchTweak tweak = SearchTweak::None)
{
    const auto& bufs = context.GetBufs().io.fwd;
    return GenericSearch<Direction::Forward>(s, context, tweak, bufs.y, bufs.x, bufs.w);
}

template <class Context>
std::shared_ptr<IPerformanceConfig> GenericSearchBwd(const GenericSearchableSolver<Context>& s,
                                                     const Context& context,
                                                     const SearchTweak tweak = SearchTweak::None)
{
    const auto& bufs = context.GetBufs().io.bwd;
    return GenericSearch<Direction::Bacward>(s, context, tweak, bufs.dx, bufs.dy, bufs.w);
}

template <class Context>
std::shared_ptr<IPerformanceConfig> GenericSearchWrW(const GenericSearchableSolver<Context>& s,
                                                     const Context& context,
                                                     const SearchTweak tweak = SearchTweak::None)
{
    const auto& bufs = context.GetBufs().io.wrw;
    return GenericSearch<Direction::Weights>(s, context, tweak, bufs.dx, bufs.dy, bufs.dw);
}

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_GENERIC_SEARCH_HPP_
