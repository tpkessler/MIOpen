#!/bin/bash

# syntax:
# miopen_gridwise_gemm_builder input_file output_file isa_version (compile-time flags)

# enable bash debugging
KMDBSCRIPT="${KMDBSCRIPT:=0}"

if [ $KMDBSCRIPT == "1" ]
then
  set -x
fi

# pass extra options to OPT
# KMOPTOPT can be used to pass last-minute options to opt in the backend
# if not set, then "-O3" would be passed to opt
KMOPTOPT="${KMOPTOPT:="-O3"}"

# pass extra options to LLC
# KMOPTLLC can be used to pass last-minute options to llc in the backend
# if not set, then "-O2" will be passed to llc
KMOPTLLC="${KMOPTLLC:="-O2"}"

# prepare env vars.
COMPILER_DIR=/opt/rocm/hcc
BIN_DIR=$COMPILER_DIR/bin
ROCM_LIB=$COMPILER_DIR/lib

CLANG=$BIN_DIR/clang
LLVM_LINK=$BIN_DIR/llvm-link
OPT=$BIN_DIR/opt
LLC=$BIN_DIR/llc
LLD=$BIN_DIR/ld.lld

# check command line arguments.
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 input_file output_file isa_version (compile-time flags)" >&2
  exit 1
fi

# parse input.
INPUT_FILE=$1
shift
OUTPUT_FILE=$1
shift
AMDGPU_TARGET=$1
shift

# launch the frontend.
$CLANG -cc1 -D__KALMAR_HC__=1 -D__HCC_HC__=1 -famp-is-device -fno-builtin -fno-common -O3 -triple amdgcn--amdhsa-hcc -aux-triple x86_64-unknown-linux-gnu -S -disable-free -disable-llvm-verifier -mrelocation-model pic -pic-level 2 -mthread-model posix -mframe-pointer=all -fmath-errno -no-integrated-as -mconstructor-aliases -fuse-init-array -target-cpu $AMDGPU_TARGET -I/opt/rocm/hcc/bin/../include -I/opt/rocm/hcc/bin/../hcc/include -isystem /opt/rocm/hip/include -isystem /opt/rocm/hsa/include -isystem /opt/rocm/hcc/include -isystem /opt/rocm/include -isystem /opt/rocm/hcc/include -isystem /opt/rocm/include -isystem /opt/rocm/hcc/include -isystem /opt/rocm/include \
-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/c++/5.4.0 \
-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/5.4.0/../../../../include/x86_64-linux-gnu/c++/5.4.0 \
-Wno-everything -Wno-unused-command-line-argument -std=c++14 -fdeprecated-macro -ferror-limit 19 -fmessage-length 0 -fgnuc-version=4.2.1 -fobjc-runtime=gcc -fcxx-exceptions -fexceptions -fdiagnostics-show-option -fcolor-diagnostics -famp -fhsa-ext -fgpu-rdc -emit-llvm-bc -o $INPUT_FILE.bc -x hc-kernel $INPUT_FILE $*

# link with rocm-device-libs
# select appropriate ROCm-Device-Libs per AMDGPU_TARGET
if [ $AMDGPU_TARGET == "gfx700" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_700.amdgcn.bc"
elif [ $AMDGPU_TARGET == "gfx701" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_701.amdgcn.bc"
elif [ $AMDGPU_TARGET == "gfx801" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_801.amdgcn.bc"
elif [ $AMDGPU_TARGET == "gfx802" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_802.amdgcn.bc"
elif [ $AMDGPU_TARGET == "gfx803" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_803.amdgcn.bc"
elif [ $AMDGPU_TARGET == "gfx900" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_900.amdgcn.bc"
elif [ $AMDGPU_TARGET == "gfx901" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_901.amdgcn.bc"
elif [ $AMDGPU_TARGET == "gfx906" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_906.amdgcn.bc"
  KMOPTLLC+=" -mattr=+sram-ecc"
elif [ $AMDGPU_TARGET == "gfx908" ]; then
  OCLC_AMDGPU_TARGET_LIB="$ROCM_LIB/oclc_isa_version_908.amdgcn.bc"
  KMOPTLLC+=" -mattr=+sram-ecc"
fi
HCC_BC_LIBS="$ROCM_LIB/hc.amdgcn.bc $ROCM_LIB/hip.amdgcn.bc $ROCM_LIB/opencl.amdgcn.bc $ROCM_LIB/ocml.amdgcn.bc $ROCM_LIB/ockl.amdgcn.bc $OCLC_ISA_VERSION_LIB $ROCM_LIB/oclc_finite_only_off.amdgcn.bc $ROCM_LIB/oclc_daz_opt_off.amdgcn.bc $ROCM_LIB/oclc_correctly_rounded_sqrt_on.amdgcn.bc $ROCM_LIB/oclc_unsafe_math_off.amdgcn.bc"

if [ -f "$ROCM_LIB/oclc_wavefrontsize64_on.amdgcn.bc" ]; then
   HCC_BC_LIBS="$HCC_BC_LIBS $ROCM_LIB/oclc_wavefrontsize64_on.amdgcn.bc"
fi

$LLVM_LINK -suppress-warnings -o $INPUT_FILE.linked.bc $INPUT_FILE.bc $HCC_BC_LIBS

# optimize bitcodes.
$OPT $KMOPTOPT -load $ROCM_LIB/LLVMSelectAcceleratorCode.so -select-accelerator-code -verify $INPUT_FILE.linked.bc -o $INPUT_FILE.opt.bc

# launch llc to produce ISA.
$LLC $KMOPTLLC -mattr=+enable-ds128 -amdgpu-enable-global-sgpr-addr -mtriple amdgcn-amd-amdhsa -mcpu=$AMDGPU_TARGET -mattr=+code-object-v3 -filetype=obj -o $INPUT_FILE.isabin $INPUT_FILE.opt.bc

# launch lld 
$LLD -shared -o $OUTPUT_FILE $INPUT_FILE.isabin
