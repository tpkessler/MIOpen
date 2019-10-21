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

#include <boost/optional.hpp>

#include <limits>
#include <vector>

namespace miopen {
namespace solver {

template <class TContext>
boost::optional<ConvSolution> SearchForSolution(const SolverBase<TContext>& solver,
                                                const TContext& ctx)
{
    ConvSolution solution;

    try
    {
        if(!solver.IsApplicable(ctx))
        {
            MIOPEN_LOG_I2(solver.DbId() << ": Not applicable");
            return boost::none;
        }

        solution = solver.GetSolution(ctx);
    }
    catch(const std::exception& exception)
    {
        MIOPEN_LOG_E("Internal error in solver " << solver.DbId() << ": " << exception.what());
        return boost::none;
    }

    if(!solution.Succeeded())
    {
        /// \todo If Solver is applicable it must provide an appropriate ConvSolution.
        /// This is not the case for some 20x5 convolutions (and possibly others).
        /// Normally we should not get here and message level should be Error.
        /// For now, let's use Info (not Warning) level to avoid
        /// flooding the console.
        MIOPEN_LOG_I(solver.DbId() << ": [Warning] Applicable Solver not succeeded.");
        return boost::none;
    }

    MIOPEN_LOG_I2(solver.DbId() << ": Success.");
    if(!solution.construction_params.empty())
        return solution;
    MIOPEN_THROW("Internal error in solver " + solver.DbId() + ": construct params are empty");
    return boost::none;
}

// Search for all applicable solutions among many solvers
template <class TContext>
std::vector<ConvSolution>
SearchForAllSolutions(const std::vector<SolverBase<TContext>*>& solvers,
                      const TContext& ctx,
                      std::size_t limit = std::numeric_limits<std::size_t>::max())
{
    std::vector<ConvSolution> ss;
    std::size_t id = 0;
    for(const auto& solver : solvers)
    {
        const auto solution = SearchForSolution(*solver, ctx);
        if(solution)
        {
            ss.push_back(*solution);
            ++id;
            if(id >= limit)
                break;
        }
    }
    return ss;
}

template <class TContext>
std::vector<std::pair<std::string, size_t>>
GetWorkspaceSize(const std::vector<SolverBase<TContext>*>& solvers, const TContext& ctx)
{
    std::vector<std::pair<std::string, size_t>> res;

    for(const auto solver : solvers)
    {
        try
        {
            if(!solver->IsApplicable(ctx))
            {
                MIOPEN_LOG_I2(solver->DbId() << ": Not applicable");
                continue;
            }

            const auto sz = solver->GetWorkspaceSize(ctx);
            res.emplace_back(solver->DbId(), sz);
        }
        catch(const std::exception& exception)
        {
            MIOPEN_LOG_E("Internal error in solver " << solver->DbId() << ": " << exception.what());
        }
    }

    return res;
}

} // namespace solver
} // namespace miopen

#endif
