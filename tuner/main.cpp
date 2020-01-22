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
#include <iostream>
#include <cstdio>

#include "conv_tuner.hpp"
#include "tuner.hpp"
#include "miopen/config.h"

int main(int argc, char* argv[])
{
    // show command
    std::cout << "MIOpenTuner:";
    for(int i = 1; i < argc; i++)
        std::cout << " " << argv[i];
    std::cout << std::endl;

    std::string base_arg = ParseBaseArg(argc, argv);

    std::unique_ptr<Tuner> tuna;
    if(base_arg == "conv")
    {
        tuna = std::make_unique<ConvTuner<float, float>>();
    }
    else if(base_arg == "convfp16")
    {
        tuna = std::make_unique<ConvTuner<float16, float>>();
    }
    else
    {
        printf("Incorrect BaseArg\n");
        exit(0);
    }

    tuna->AddCmdLineArgs();
    tuna->ParseCmdLineArgs(argc, argv);
    tuna->GetandSetData();
    tuna->AllocateBuffersAndCopy();

    int fargval = tuna->GetInputFlags().GetValueInt("forw");

    //int iter = tuna->GetInputFlags().GetValueInt("iter");
    int status;

    if(fargval & 1 || fargval == 0)
    {
        status = tuna->RunForwardGPU();
    }

    if(fargval != 1)
    {
        status = tuna->RunBackwardGPU();
    }

    return 0;
}
