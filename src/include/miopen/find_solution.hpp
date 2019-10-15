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

#ifndef MIOPEN_GUARD_MLOPEN_FIND_SOLUTION_HPP
#define MIOPEN_GUARD_MLOPEN_FIND_SOLUTION_HPP

#include <miopen/env.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/solver.hpp>

#include <vector>

/// Allows to explicitly disable performance filtering heuristics
/// in "Find first convolution only" mode.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING)

namespace miopen {
namespace solver {

template <class TContext>
ConvSolution SearchForSolution(const std::vector<SolverBase<TContext>*>& solvers,
                               const TContext& search_params)
{
// Using const here causes gcc to ICE
#if(!defined(__GNUC__) || defined(__clang__))
    const
#endif
        auto no_perf_filtering = miopen::IsDisabled(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING{});

    for(const auto solver : solvers)
    {
        if(!solver->IsApplicable(search_params) ||
           !(no_perf_filtering || solver->IsFast(search_params)))
        {
            MIOPEN_LOG_I2(solver->DbId() << ": Not applicable");
            continue;
        }

        const auto solution = solver->GetSolution(search_params);
        if(!solution.Succeeded())
            continue;
        MIOPEN_LOG_I2(solver->DbId() << ": Success.");
        if(!solution.construction_params.empty())
            return solution;
        MIOPEN_THROW(std::string("Internal error in solver: ") + solver->DbId());
    }

    return ConvSolution{miopenStatusUnknownError};
}

// Search for all applicable solutions among many solvers
template <class TContext>
std::vector<ConvSolution> SearchForAllSolutions(const std::vector<SolverBase<TContext>*>& solvers,
                                                const TContext& search_params)
{
    std::vector<ConvSolution> ss;
    for(const auto& solver : solvers)
    {
        if(!solver->IsApplicable(search_params))
        {
            MIOPEN_LOG_I2(solver->DbId() << ": Not applicable");
            continue;
        }

        const auto s = solver->GetSolution(search_params);
        if(s.Succeeded())
        {
            ss.push_back(s);
            MIOPEN_LOG_I2(solver->DbId() << ": Success.");
            continue;
        }

        /// \todo If Solver is applicable it must provide an appropriate ConvSolution.
        /// This is not the case for some 20x5 convolutions (and possibly others).
        /// Normally we should not get here and message level should be Error.
        /// For now, let's use Info (not Warning) level to avoid
        /// flooding the console.
        MIOPEN_LOG_I(solver->DbId() << ": [Warning] Applicable Solver not succeeded.");
    }
    return ss;
}

template <class TContext>
std::vector<std::pair<std::string, size_t>>
GetWorkspaceSize(const std::vector<SolverBase<TContext>*>& solvers, const TContext& search_params)
{
    std::vector<std::pair<std::string, size_t>> res;

    for(const auto solver : solvers)
    {
        if(!solver->IsApplicable(search_params))
        {
            MIOPEN_LOG_I2(solver->DbId() << ": Not applicable");
            continue;
        }

        const auto sz = solver->GetWorkspaceSize(search_params);
        res.emplace_back(solver->DbId(), sz);
    }

	return res;
}

} // namespace solver
} // namespace miopen

#endif
