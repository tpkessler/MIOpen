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

#include <miopen/config.h>
#include <miopen/hip_build_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/exec_utils.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <boost/optional.hpp>
#include <sstream>
#include <string>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_HIP_ENFORCE_COV3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_HIP_VERBOSE)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_HIP_DUMP)

namespace miopen {

bool IsHccCompiler()
{
    static const auto isHcc = EndsWith(MIOPEN_HIP_COMPILER, "hcc");
    return isHcc;
}

bool IsHipClangCompiler()
{
    static const auto isClangXX = EndsWith(MIOPEN_HIP_COMPILER, "clang++");
    return isClangXX;
}

namespace {

inline bool ProduceCoV3()
{
    // If env.var is set, then let's follow it.
    if(IsEnabled(MIOPEN_DEBUG_HIP_ENFORCE_COV3{}))
        return true;
    if(IsDisabled(MIOPEN_DEBUG_HIP_ENFORCE_COV3{}))
        return false;
    // Otherwise, let's enable CO v3 for HIP kernels since ROCm 3.0.
    return (HipCompilerVersion() >= external_tool_version_t{3, 0, -1});
}

/// Returns option for enabling/disabling CO v3 generation for the compiler
/// that builds HIP kernels, depending on compiler version etc.
inline const std::string& GetCoV3Option(const bool enable)
{
    /// \note PR #2166 uses the "--hcc-cov3" option when isHCC is true.
    /// It's unclear why... HCC included in ROCm 2.8 does not support it,
    /// perhaps it suits for some older HCC?
    ///
    /// These options are Ok for ROCm 3.0:
    static const std::string option_enable{"-mcode-object-v3"};
    static const std::string no_option{};
    if(enable)
        return option_enable;
    else
        return no_option;
}
} // namespace

boost::filesystem::path HipBuild(boost::optional<TmpDir>& tmp_dir,
                                 const std::string& filename,
                                 std::string src,
                                 std::string params,
                                 const std::string& dev_name,
                                 const std::string& extra_options)
{
#ifdef __linux__
    MIOPEN_LOG_I("filename: " << filename);
    MIOPEN_LOG_I("src: " << src);
    MIOPEN_LOG_I("params: " << params);
    MIOPEN_LOG_I("dev_name: " << dev_name);
    MIOPEN_LOG_I("extra_options: " << extra_options);

    // write out the include files
    auto inc_list = GetKernelIncList();
    auto inc_path = tmp_dir->path;
    boost::filesystem::create_directories(inc_path);
    for(auto inc_file : inc_list)
    {
        auto inc_src = GetKernelInc(inc_file);
        WriteFile(inc_src, inc_path / inc_file);
    }
    auto input_file = tmp_dir->path / filename;
    auto bin_file = tmp_dir->path / (filename + ".o");

    // invoke mlir kernel generator.
    auto mlir_file = tmp_dir->path / "gridwise_convolution_implicit_gemm_v4r4_mlir";
    MIOPEN_LOG_I("invoke MLIR kernel generator.");
    MIOPEN_LOG_I("C++ source: " << mlir_file.string() << ".cpp");
    MIOPEN_LOG_I("C++ header: " << mlir_file.string() << ".hpp");
    // --p=false to disable MLIR default value population
    tmp_dir->Execute("/opt/rocm/miopen/bin/miopen_mlir_generator.sh",
                     mlir_file.string() + " " + extra_options + " --p=false");

    // get mlir kernel compilation flags.
    auto mlir_cflags_file = tmp_dir->path / "gridwise_convolution_implicit_gemm_v4r4_mlir_cflags";
    MIOPEN_LOG_I("getting MLIR kernel cflags.");
    // --p=false to disable MLIR default value population
    tmp_dir->Execute("/opt/rocm/miopen/bin/miopen_mlir_cflags.sh",
                     mlir_cflags_file.string() + " " + extra_options + " --p=false");

    if (!boost::filesystem::exists(mlir_cflags_file))
        MIOPEN_THROW(filename + " failed to build due to missing compile-time flags");

    std::string cflags;
    cflags.reserve(4096);
    bin_file_to_str(mlir_cflags_file, cflags);

    // skip first line
    cflags = cflags.substr(cflags.find("\n") + 1);
    // compile
    MIOPEN_LOG_I("input_file: " << input_file.string());
    MIOPEN_LOG_I("output_file: " << bin_file.string());
    MIOPEN_LOG_I("isa: " << dev_name);
    MIOPEN_LOG_I("params: " << cflags);
    tmp_dir->Execute("/opt/rocm/miopen/bin/miopen_gridwise_gemm_builder.sh",
                     input_file.string() + " " +
                     bin_file.string() + "  " +
                     dev_name + " " +
                     cflags
                     );
    if(!boost::filesystem::exists(bin_file))
        MIOPEN_THROW(filename + " failed to compile");
//#ifdef EXTRACTKERNEL_BIN
//    if(IsHccCompiler())
//    {
//        // call extract kernel
//        tmp_dir->Execute(EXTRACTKERNEL_BIN, " -i " + bin_file.string());
//        auto hsaco =
//            std::find_if(boost::filesystem::directory_iterator{tmp_dir->path},
//                         {},
//                         [](auto entry) { return (entry.path().extension() == ".hsaco"); });
//
//        if(hsaco == boost::filesystem::directory_iterator{})
//        {
//            MIOPEN_LOG_E("failed to find *.hsaco in " << hsaco->path().string());
//        }
//
//        return hsaco->path();
//    }
//    else
//#endif
//#ifdef MIOPEN_OFFLOADBUNDLER_BIN
//        // clang-format off
//    if(IsHipClangCompiler())
//    {
//        // clang-format on
//
//        // call clang-offload-bundler
//        tmp_dir->Execute(MIOPEN_OFFLOADBUNDLER_BIN,
//                         "--type=o --targets=hip-amdgcn-amd-amdhsa-" + dev_name + " --inputs=" +
//                             bin_file.string() + " --outputs=" + bin_file.string() +
//                             ".hsaco --unbundle");
//
//        auto hsaco =
//            std::find_if(boost::filesystem::directory_iterator{tmp_dir->path},
//                         {},
//                         [](auto entry) { return (entry.path().extension() == ".hsaco"); });
//
//        if(hsaco == boost::filesystem::directory_iterator{})
//        {
//            MIOPEN_LOG_E("failed to find *.hsaco in " << hsaco->path().string());
//        }
//        return hsaco->path();
//    }
//    else
//#endif
//    {
        return bin_file;
//    }
#else
    (void)filename;
    (void)params;
    MIOPEN_THROW("HIP kernels are only supported in Linux");
#endif
}

void bin_file_to_str(const boost::filesystem::path& file, std::string& buf)
{
    std::ifstream bin_file_ptr(file.string().c_str(), std::ios::binary);
    std::ostringstream bin_file_strm;
    bin_file_strm << bin_file_ptr.rdbuf();
    buf = bin_file_strm.str();
}

static external_tool_version_t HipCompilerVersionImpl()
{
    external_tool_version_t version;
    if(IsHccCompiler())
    {
        const std::string path(MIOPEN_HIP_COMPILER);
        const std::string mandatory_prefix("(based on HCC ");
        do
        {
            if(path.empty() || !std::ifstream(path).good())
                break;

            std::stringstream out;
            MIOPEN_LOG_NQI2("Running: " << '\'' << path << " --version" << '\'');
            if(miopen::exec::Run(path + " --version", nullptr, &out) != 0)
                break;

            std::string line;
            while(!out.eof())
            {
                std::getline(out, line);
                MIOPEN_LOG_NQI2(line);
                auto begin = line.find(mandatory_prefix);
                if(begin == std::string::npos)
                    continue;

                begin += mandatory_prefix.size();
                int v3, v2, v1 = v2 = v3 = -1;
                char c2, c1 = c2 = 'X';
                std::istringstream iss(line.substr(begin));
                iss >> v1 >> c1 >> v2 >> c2 >> v3;
                if(!iss.fail() && v1 >= 0)
                {
                    version.major = v1;
                    if(c1 == '.' && v2 >= 0)
                    {
                        version.minor = v2;
                        if(c2 == '.' && v3 >= 0)
                            version.patch = v3;
                    }
                }
                break;
            }
        } while(false);
    }
    else
    {
#ifdef HIP_PACKAGE_VERSION_MAJOR
        MIOPEN_LOG_NQI2("Read version information from HIP package...");
        version.major = HIP_PACKAGE_VERSION_MAJOR;
#ifdef HIP_PACKAGE_VERSION_MINOR
        version.minor = HIP_PACKAGE_VERSION_MINOR;
#else
        version.minor = 0;
#endif
#ifdef HIP_PACKAGE_VERSION_PATCH
        version.patch = HIP_PACKAGE_VERSION_PATCH;
#else
        version.patch = 0;
#endif
#else // HIP_PACKAGE_VERSION_MAJOR is not defined. CMake failed to find HIP package.
        MIOPEN_LOG_NQI2("...assuming 3.2.0 (hip-clang RC)");
        version.major = 3;
        version.minor = 2;
        version.patch = 0;
#endif
    }
    MIOPEN_LOG_NQI(version.major << '.' << version.minor << '.' << version.patch);
    return version;
}

external_tool_version_t HipCompilerVersion()
{
    static auto once = HipCompilerVersionImpl();
    return once;
}

bool external_tool_version_t::operator>(const external_tool_version_t& rhs) const
{
    if(major > rhs.major)
        return true;
    else if(major == rhs.major)
    {
        if(minor > rhs.minor)
            return true;
        else if(minor == rhs.minor)
            return (patch > rhs.patch);
        else
            return false;
    }
    else
        return false;
}

bool external_tool_version_t::operator>=(const external_tool_version_t& rhs) const
{
    if(major > rhs.major)
        return true;
    else if(major == rhs.major)
    {
        if(minor > rhs.minor)
            return true;
        else if(minor == rhs.minor)
            return (patch >= rhs.patch);
        else
            return false;
    }
    else
        return false;
}

} // namespace miopen
