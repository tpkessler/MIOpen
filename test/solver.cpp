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

#include <miopen/convolution.hpp>
#include <miopen/db.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/solver.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/temp_file.hpp>

#include <cstdlib>
#include <functional>
#include <sstream>
#include <typeinfo>

#include "get_handle.hpp"
#include "test.hpp"

namespace miopen {
namespace tests {
class TrivialTestSolver final : public solver::SolverBase<ConvolutionContext>
{
    public:
    static const char* FileName() { return "TrivialTestSolver"; }
    const std::string& DbId() const override { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& context) const override
    {
        return context.in_width == 1;
    }

    solver::ConvSolution GetSolution(const ConvolutionContext&) const override
    {
        solver::ConvSolution ret;
        solver::KernelInfo kernel;

        kernel.kernel_file  = FileName();
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }
};

struct TestConfig final : solver::Serializable<TestConfig>, IPerformanceConfig
{
    std::string str;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.str, "str");
    }

    bool SetNextValue() override
    {
        MIOPEN_THROW("TestConfig doesn't support generic_search");
        return false;
    };
    bool IsValid(const ConvolutionContext&) const override
    {
        MIOPEN_THROW("TestConfig doesn't support generic_search");
        return false;
    };
    bool operator==(const IPerformanceConfig&) const override
    {
        MIOPEN_THROW("TestConfig doesn't support generic_search");
        return false;
    };
};

class SearchableTestSolver final : public solver::SearchableSolver<ConvolutionContext>
{
    public:
    static int searches_done() { return _serches_done; }
    static const char* FileName() { return "SearchableTestSolver"; }
    static const char* NoSearchFileName() { return "SearchableTestSolver.NoSearch"; }
    const std::string& DbId() const final { return SolverDbId(*this); }
    bool IsApplicable(const ConvolutionContext& /* context */) const final { return true; }

    std::shared_ptr<IPerformanceConfig> GetPerformanceConfig(const ConvolutionContext&) const final
    {
        TestConfig config{};
        config.str = NoSearchFileName();
        return std::make_shared<TestConfig>(config);
    }

    bool IsValidPerformanceConfig(const ConvolutionContext&, const IPerformanceConfig&) const final
    {
        return true;
    }

    std::shared_ptr<IPerformanceConfig> Search(const ConvolutionContext&) const final
    {
        TestConfig config;
        config.str = FileName();
        _serches_done++;
        return std::make_shared<TestConfig>(config);
    }

    solver::ConvSolution
    GetSolution(const ConvolutionContext&, const IPerformanceConfig& config_, bool) const final
    {
        auto config = dynamic_cast<const TestConfig&>(config_);
        solver::ConvSolution ret;
        solver::KernelInfo kernel;

        kernel.kernel_file  = config.str;
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }

    private:
    static int _serches_done;

    std::unique_ptr<IPerformanceConfig> AllocateConfig() const final
    {
        return std::make_unique<TestConfig>();
    }
};

int SearchableTestSolver::_serches_done = 0;

static solver::ConvSolution FindSolution(const ConvolutionContext& ctx, const std::string& db_path)
{
    test::db_path_override() = db_path;

    static const auto solvers =
        std::vector<ConvSolver*>{&StaticContainer<TrivialTestSolver>::Instance(),
                                 &StaticContainer<SearchableTestSolver>::Instance()};

    return SearchForAllSolutions(solvers, ctx, 1).front();
}

class SolverTest
{
    public:
    void Run() const
    {
        const TempFile db_path("miopen.tests.solver");

        ConstructTest(db_path, TrivialTestSolver::FileName(), {0, 0, 0, 1});

        ConstructTest(db_path,
                      TrivialTestSolver::FileName(),
                      {0, 0, 0, 1},
                      [](ConvolutionContext& c) { c.do_search = true; });

        ConstructTest(db_path,
                      SearchableTestSolver::NoSearchFileName(),
                      {0, 0, 0, 0},
                      [](ConvolutionContext& c) { c.do_search = false; });

        ConstructTest(db_path,
                      SearchableTestSolver::FileName(),
                      {0, 0, 0, 0},
                      [](ConvolutionContext& c) { c.do_search = true; });

        const auto& searchable_solver = StaticContainer<const SearchableTestSolver>::Instance();
        const auto searches           = SearchableTestSolver::searches_done();

        // Should read in both cases: result is already in DB, solver is searchable.
        ConstructTest(
            db_path, SearchableTestSolver::FileName(), {0, 0, 0, 0}, [](ConvolutionContext&) {});

        ConstructTest(db_path,
                      SearchableTestSolver::FileName(),
                      {0, 0, 0, 0},
                      [](ConvolutionContext& c) { c.do_search = true; });

        // Checking no more searches were done.
        EXPECT_EQUAL(searches, searchable_solver.searches_done());
    }

    private:
    static void ConstructTest(const std::string& db_path,
                              const char* expected_kernel,
                              const std::initializer_list<size_t>& in,
                              const std::function<void(ConvolutionContext&)>& context_filler =
                                  [](ConvolutionContext&) {})
    {
        auto ctx = ConvolutionContext{TensorDescriptor{miopenFloat, in},
                                      TensorDescriptor{},
                                      TensorDescriptor{},
                                      ConvolutionDescriptor{},
                                      1};
        ctx.SetStream(&get_handle());
        context_filler(ctx);

        const auto sol = FindSolution(ctx, db_path);

        EXPECT_OP(sol.construction_params.size(), >, 0);
        EXPECT_EQUAL(sol.construction_params[0].kernel_file, expected_kernel);
    }
};
} // namespace tests
} // namespace miopen

int main() { miopen::tests::SolverTest().Run(); }
