def rocmnode(name) {
    return 'rocmtest && miopen && ' + name
}


def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        cat /sys/module/amdgpu/version
        ls /opt/ -la
    """
}

//default
// CXX=/opt/rocm/llvm/bin/clang++ CXXFLAGS='-Werror' cmake -DMIOPEN_GPU_SYNC=Off -DCMAKE_PREFIX_PATH=/usr/local -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release ..
//
def cmake_build(Map conf=[:]){

    def compiler = conf.get("compiler","/opt/rocm/llvm/bin/clang++")
    def config_targets = conf.get("config_targets","check")
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined " + conf.get("extradebugflags", "")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/usr/local")
    def setup_args = " -DMIOPEN_GPU_SYNC=Off " + conf.get("setup_flags","")

    if (prefixpath != "/usr/local"){
        setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "
    }

    def build_type_debug = (conf.get("build_type",'release') == 'debug')

    //cmake_env can overwrite default CXX variables.
    def cmake_envs = "CXX=${compiler} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    def package_build = (conf.get("package_build","") == "true")

    if (package_build == true) {
        config_targets = "package"
    }

    if(conf.get("build_install","") == "true")
    {
        config_targets = 'install ' + config_targets
        setup_args = ' -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install' + setup_args
    } else{
        setup_args = ' -DBUILD_DEV=On' + setup_args
    }

    // test_flags = ctest -> MIopen flags
    def test_flags = conf.get("test_flags","")

    if (conf.get("vcache_enable","") == "true"){
        def vcache = conf.get(vcache_path,"/var/jenkins/.cache/miopen/vcache")
        build_envs = " MIOPEN_VERIFY_CACHE_PATH='${vcache}' " + build_envs
    } else{
        test_flags = " --disable-verification-cache " + test_flags
    }

    if(conf.get("codecov", false)){ //Need
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags} -fprofile-arcs -ftest-coverage' -DCODECOV_TEST=On " + setup_args
    }else if(build_type_debug){
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'" + setup_args
    }else{
        setup_args = " -DCMAKE_BUILD_TYPE=release" + setup_args
    }

    if(test_flags != ""){
       setup_args = "-DMIOPEN_TEST_FLAGS='${test_flags}'" + setup_args
    }

    def pre_setup_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            rm -rf install
            mkdir install
            rm -f src/kernels/*.ufdb.txt
            rm -f src/kernels/miopen*.udb
            cd build
        """
    def setup_cmd = conf.get("setup_cmd", "${cmake_envs} cmake ${setup_args}   .. ")
    def build_cmd = conf.get("build_cmd", "${build_envs} dumb-init make -j\$(nproc) ${config_targets}")
    def execute_cmd = conf.get("execute_cmd", "")

    def cmd = conf.get("cmd", """
            ${pre_setup_cmd}
            ${setup_cmd}
            ${build_cmd}
            ${execute_cmd}
        """)

    echo cmd
    sh cmd

    // Only archive from master or develop
    if (package_build == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        env.CODECOV_TOKEN="aec031be-7673-43b5-9840-d8fb71a2354e"
        checkout scm

        def image = "miopen"
        def prefixpath = conf.get("prefixpath", "/usr/local")
        def gpu_arch = conf.get("gpu_arch", "gfx900;gfx906")

        def miotensile_version = conf.get("miotensile_version", "default")
        def target_id = conf.get("target_id", "OFF")
        def mlir_build = conf.get("mlir_build", "OFF")
        def build_fin = conf.get("build_fin", "OFF")
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg MIOTENSILE_VER='${miotensile_version}' --build-arg USE_TARGETID='${target_id}' --build-arg USE_MLIR='${mlir_build}' --build-arg USE_FIN='${build_fin}' "

        def variant = env.STAGE_NAME

        def codecov = conf.get("codecov", false)

        def retimage
        gitStatusWrapper(credentialsId: '7126e5fe-eb51-4576-b52b-9aaf1de8f0fd', gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'MIOpen') {
            try {
                retimage = docker.build("${image}", dockerArgs + '.')
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES')
                    {
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            catch(Exception ex) {
                retimage = docker.build("${image}", dockerArgs + "--no-cache .")
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES')
                    {
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                    }
                }
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 5, unit: 'HOURS')
                {
                    cmake_build(conf)

                    if (codecov) {
                        sh '''
                            cd build
                            lcov --directory . --capture --output-file $(pwd)/coverage.info
                            lcov --remove $(pwd)/coverage.info '/usr/*' --output-file $(pwd)/coverage.info
                            lcov --list $(pwd)/coverage.info
                            curl -s https://codecov.io/bash | bash
                            echo "Uploaded"
                        '''
                    }
                }
            }
        }
        return retimage
}

def reboot(){
    build job: 'reboot-slaves', propagate: false , parameters: [string(name: 'server', value: "${env.NODE_NAME}"),]
}

def buildHipClangJobAndReboot(Map conf=[:]){
    try{
        buildHipClangJob(conf)
    }
    catch(e){
        echo "throwing error exception for the stage"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (!conf.get("no_reboot", false)) {
            reboot()
        }
    }
}


/// Stage name format:
/// [DataType] Backend[/Compiler] BuildType [TestSet] [Target]
///
/// The only mandatory elements are Backend and BuildType; others are optional.
///
/// DataType := { Fp16 | Bf16 | Int8 | Fp32 }
/// Backend := { Hip | OpenCL | HipNoGPU}
/// Compiler := { Clang* | GCC* }
///   * "Clang" is the default for the Hip backend, and implies hip-clang compiler.
///     For the OpenCL backend, "Clang" implies the system x86 compiler.
///   * "GCC" is the default for OpenCL backend.
///   * The default compiler is usually not specified.
/// BuildType := { Release* | Debug | Install } [ BuildTypeModifier ]
///   * BuildTypeModifier := { COMGR | Embedded | Static | Normal-Find | Fast-Find
///                            MLIR | Tensile | Tensile-Latest | Package | ... }
/// TestSet := { All | Smoke* } [ Codecov ]
///   * "All" corresponds to "cmake -DMIOPEN_TEST_ALL=On".
///   * "Smoke" (-DMIOPEN_TEST_ALL=Off) is the default and usually not specified.
///   * "Codecov" is optional code coverage analysis.
/// Target := { gfx908 | Vega20 | Vega10 | Vega* | gfx1030 }
///   * "Vega" (gfx906 or gfx900) is the default and usually not specified.


pipeline {
    agent none
    options {
        parallelsAlwaysFailFast()
    }
    parameters {
        booleanParam(
            name: "DISABLE_ALL_STAGES",
            defaultValue: false,
            description: "Disables each stage in the pipline")
        booleanParam(
            name: "STATIC_CHECKS",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "SMOKE_FP32_AUX1",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "SMOKE_FP16_BF16_INT8",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "SMOKE_MLIR",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "SMOKE_MIOPENTENSILE_LATEST",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "FULL_TESTS",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "MIOPENTENSILE",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "MIOPENTENSILE_LATEST",
            defaultValue: false,
            description: "")
        booleanParam(
            name: "PACKAGES",
            defaultValue: true,
            description: "")
        booleanParam(
                name: "BUILD_PACKAGES",
                defaultValue: true,
                description: "Run packages stage")
    }

    environment{
        extra_log_env   = " MIOPEN_LOG_LEVEL=5 "
        Fp16_flags      = " -DMIOPEN_TEST_HALF=On"
        Bf16_flags      = " -DMIOPEN_TEST_BFLOAT16=On"
        Int8_flags      = " -DMIOPEN_TEST_INT8=On"
        Full_test       = " -DMIOPEN_TEST_ALL=On"
        Tensile_build_env = "MIOPEN_DEBUG_HIP_KERNELS=0 "
        Tensile_setup = " -DMIOPEN_TEST_MIOTENSILE=ON -DMIOPEN_USE_MIOPENTENSILE=ON -DMIOPEN_USE_ROCBLAS=OFF"
    }
    stages{
        stage("Static checks"){
            when { expression { params.STATIC_CHECKS && !params.DISABLE_ALL_STAGES } }
            parallel{
                stage('Hip Tidy') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        setup_cmd = "CXX='/opt/rocm/llvm/bin/clang++' cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On .. "
                        build_cmd = "make -j\$(nproc) -k analyze"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd, no_reboot:true)
                    }
                }
                stage('OpenCL Tidy') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        setup_cmd = "cmake -DMIOPEN_BACKEND=OpenCL -DBUILD_DEV=On .."
                        build_cmd = "make -j\$(nproc) -k analyze"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: setup_cmd, build_cmd: build_cmd, no_reboot:true)
                    }
                }
                stage('Clang Format') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        execute_cmd = "find . -iname \'*.h\' \
                                -o -iname \'*.hpp\' \
                                -o -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-10 -style=file {} | diff - {}\'"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, no_reboot:true)
                    }
                }
              stage('Tuna Fin Build Test') {
                  agent{ label rocmnode("nogpu") }
                  environment{
                      setup_cmd = "CXX='/opt/rocm/llvm/bin/clang++' cmake -DCMAKE_BUILD_TYPE=DEBUG -DMIOPEN_BACKEND=HIPNOGPU -DBUILD_SHARED_LIBS=Off -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_ENABLE_FIN=ON .. "
                      build_cmd = "make -j\$(nproc) "
                  }
                  steps{
                      buildHipClangJobAndReboot(setup_cmd: setup_cmd, execute_cmd: "", no_reboot:true, build_cmd: build_cmd, build_fin: "ON")
                  }
              }
            }
        }
        stage("Smoke Fp32"){
            when { expression { params.SMOKE_FP32_AUX1 && !params.DISABLE_ALL_STAGES } }
            environment{
                Smoke_targets = "check doc MIOpenDriver"
            }
            parallel{
               stage('Fp32 OpenCL Debug + Codecov') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', build_type: 'debug', config_targets: Smoke_targets, codecov: true)
                    }
                }
                stage('Fp32 OpenCL gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', config_targets: Smoke_targets, gpu_arch: "gfx908")
                    }
                }
                stage('Fp32 Hip /opt/rocm') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(prefixpath: '/opt/rocm', config_targets: Smoke_targets)
                    }
                }
                stage('Fp32 Hip Debug') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(build_type: 'debug', config_targets: Smoke_targets)
                    }
                }
                stage('Fp32 OpenCL Debug gfx1030') {
                    agent{ label rocmnode("navi21") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', build_type: 'debug', config_targets: Smoke_targets, build_env: extra_log_env, gpu_arch: "gfx1030")
                    }
                }
                stage('Fp32 Hip Debug gfx908 /opt/rocm') {
                    agent{ label rocmnode("gfx908") }
                    environment{
                        gpu_arch = "gfx908"
                        prefixpath = "/opt/rocm"
                    }
                    steps{
                        buildHipClangJobAndReboot(prefixpath: prefixpath, build_type: 'debug', config_targets: Smoke_targets, gpu_arch: gpu_arch)
                    }
                }
                stage('Fp32 HipNoGPU Debug') {
                    agent{  label rocmnode("nogpu") }
                    environment{
                        HipNoGPU_flags = "-DMIOPEN_BACKEND=HIPNOGPU -DMIOPEN_INSTALL_CXX_HEADERS=On"
                        build_cmd = "make -j\$(nproc)"
                    }
                    steps{
                        buildHipClangJob( build_type: 'debug', setup_flags: HipNoGPU_flags, build_cmd: build_cmd)
                    }
                }
            }
        }
        stage("Smoke Aux 1"){
            when { expression { params.SMOKE_FP32_AUX1 && !params.DISABLE_ALL_STAGES } }
            parallel{
                stage('Fp32 Hip Debug COMGR') {
                    agent{ label rocmnode("vega") }
                    environment{
                        COMGR_build_cmd = "CTEST_PARALLEL_LEVEL=2 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 MIOPEN_LOG_LEVEL=5 make -j\$(nproc) check"
                    }
                    steps{
                        buildHipClangJobAndReboot( build_type: 'debug', setup_flags: "-DMIOPEN_USE_COMGR=On", build_cmd: COMGR_build_cmd, test_flags: ' --verbose ')
                    }
                }
                stage('Fp32 Hip Debug Embedded Vega20') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        Embedded_flags = "-DMIOPEN_EMBED_DB='gfx906_60;gfx906_64'"
                    }
                    steps{
                        buildHipClangJobAndReboot( build_type: 'debug', setup_flags: Embedded_flags, build_env: extra_log_env, test_flags: ' --verbose ')
                    }
                }
                stage('Fp32 Hip Static') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: "-DBUILD_SHARED_LIBS=Off")
                    }
                }
                stage('Fp32 Hip Normal-Find') {
                    agent{ label rocmnode("vega") }
                    environment{
                        config_targets = "test_conv2d"
                        execute_cmd = "MIOPEN_FIND_MODE=1 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot(config_targets: config_targets, execute_cmd: execute_cmd)
                    }
                }
                stage('Fp32 Hip Fast-Find') {
                    agent{ label rocmnode("vega") }
                    environment{
                        config_targets =   "test_conv2d"
                        execute_cmd = "MIOPEN_FIND_MODE=2 CTEST_PARALLEL_LEVEL=4  MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 bin/test_conv2d --disable-verification-cache"
                    }
                    steps{
                        buildHipClangJobAndReboot( config_targets: config_targets, execute_cmd: execute_cmd)
                    }
                }
                stage('Fp32 Hip') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot()
                    }
                }
            }
        }
        stage("Smoke MLIR"){
            when { expression { params.SMOKE_MLIR && !params.DISABLE_ALL_STAGES } }
            environment{ MLIR_flags = "-DMIOPEN_USE_MLIR=On" }
            parallel{
                stage('Fp32 Hip MLIR') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: MLIR_flags, build_env: extra_log_env, test_flags: ' --verbose ', mlir_build: "ON")
                    }
                }
                stage('Fp16 Hip MLIR') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: MLIR_flags + Fp16_flags, build_env: extra_log_env, test_flags: ' --verbose ', mlir_build: "ON")
                    }
                }
                stage('Fp16 Hip MLIR gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: MLIR_flags + Fp16_flags, build_env: extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908", mlir_build: "ON")
                    }
                }
                stage('Fp32 Hip MLIR gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: MLIR_flags, build_env: extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908", mlir_build: "ON")
                    }
                }
            }
        }
        stage("Smoke Fp16/Bf16/Int8"){
            when { expression { params.SMOKE_FP16_BF16_INT8 && !params.DISABLE_ALL_STAGES } }
            environment{
                Smoke_targets = "check doc MIOpenDriver"
            }
            parallel{
                stage('Fp16 OpenCL Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Fp16_flags, config_targets: Smoke_targets)
                    }
                }
                stage('Int8 OpenCL Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Int8_flags, config_targets: Smoke_targets)
                    }
                }
                stage('Fp16 Hip Vega20 /opt/rocm') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets)
                    }
                }
                stage('Bf16 Hip Vega20 /opt/rocm') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets)
                    }
                }
                stage('Fp16 Hip gfx908 /opt/rocm') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Fp16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets, gpu_arch: "gfx908")
                    }
                }
                stage('Bf16 Hip gfx908 /opt/rocm') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags, prefixpath: '/opt/rocm', config_targets: Smoke_targets, gpu_arch: "gfx908")
                    }
                }
            }
        }
        stage("Smoke MIOpenTensile Latest"){
            environment{
                Tensile_version = "latest"
            }
            parallel{
                stage('Fp16 Hip Tensile-Latest All Vega20') {
                    when { expression { params.SMOKE_MIOPENTENSILE_LATEST && !params.DISABLE_ALL_STAGES } }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Fp16_flags, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Int8 Hip Tensile-Latest All Vega20') {
                    when { expression { params.SMOKE_MIOPENTENSILE_LATEST && !params.DISABLE_ALL_STAGES } }
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + extra_log_env + Int8_flags, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }

                stage('Fp32 Hip Tensile-Latest All gfx908') {
                    when { expression { params.SMOKE_MIOPENTENSILE_LATEST && !params.DISABLE_ALL_STAGES } }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + extra_log_env, build_env: Tensile_build_env + extra_log_env, gpu_arch: "gfx908:xnack-", test_flags: ' --verbose ', miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Bf16 Hip Tensile-Latest All gfx908') {
                    when { expression { params.SMOKE_MIOPENTENSILE_LATEST && !params.DISABLE_ALL_STAGES } }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + extra_log_env + Bf16_flags, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
            }
        }
        stage("Full Tests I"){
            when { expression { params.FULL_TESTS && !params.DISABLE_ALL_STAGES } }
            parallel{
                stage('Int8 HIP All Vega20 /opt/rocm') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Int8_flags + Full_test, prefixpath: '/opt/rocm')
                    }
                }
                stage('Fp32 OpenCL Install All') {
                    agent{ label rocmnode("vega") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Full_test, build_install: "true")
                    }
                }
                stage('Bf16 Hip Install All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Bf16_flags + Full_test, build_install: "true", gpu_arch: "gfx908")
                    }
                }
            }
        }

        stage("Full Tests II"){
            when { expression { params.FULL_TESTS && !params.DISABLE_ALL_STAGES } }
            environment{
                WORKAROUND_iGemm_936 = " MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0"
            }
            parallel{
                stage('Fp32 Hip All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test, build_env: WORKAROUND_iGemm_936, gpu_arch: "gfx908")
                    }
                }
                stage('Fp16 Hip Install All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Full_test + Fp16_flags, build_env: WORKAROUND_iGemm_936, build_install: "true")
                    }
                }
                stage('Fp32 Hip All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Full_test, build_env: WORKAROUND_iGemm_936)
                    }
                }
                stage('Fp32 OpenCL All gfx1030') {
                    agent{ label rocmnode("navi21") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', setup_flags: Full_test, build_env: extra_log_env, gpu_arch: "gfx1030")
                    }
                }
                stage('Fp16 Hip All Install gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_env: WORKAROUND_iGemm_936, build_install: "true", gpu_arch: "gfx908")
                    }
                }
            }
        }

        stage("MIOpenTensile"){
            when { expression { params.MIOPENTENSILE && !params.DISABLE_ALL_STAGES } }
            environment{
                Tensile_version = "default"
            }
            parallel{
                stage('Fp32 Hip Tensile All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Fp16 Hip Tensile All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + Fp16_flags, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Bf16 Hip Tensile All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Bf16_flags, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Int8 Hip Tensile All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Int8_flags, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Fp32 Hip Tensile All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Fp16 Hip Tensile All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Fp16_flags, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Bf16 Hip Tensile All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Bf16_flags, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Int8 Hip Tensile All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Int8_flags, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
            }
        }
        stage("MIOpenTensile Latest"){
            when { expression { params.MIOPENTENSILE_LATEST && !params.DISABLE_ALL_STAGES  } }
            environment{
                Tensile_version = "latest"
            }
            parallel{
                stage('Fp32 Hip Tensile-Latest All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Fp16 Hip Tensile-Latest All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + Fp16_flags, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Bf16 Hip Tensile-Latest All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Bf16_flags, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Int8 Hip Tensile-Latest All Vega20') {
                    agent{ label rocmnode("vega20") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Int8_flags, build_env: Tensile_build_env, test_flags: ' --verbose ', gpu_arch:"gfx906:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Fp32 Hip Tensile-Latest All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Fp16 Hip Tensile-Latest All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Fp16_flags, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Bf16 Hip Tensile-Latest All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Bf16_flags, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
                stage('Int8 Hip Tensile-Latest All gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot( setup_flags: Tensile_setup + Full_test + extra_log_env + Int8_flags, build_env: Tensile_build_env + extra_log_env, test_flags: ' --verbose ', gpu_arch: "gfx908:xnack-", miotensile_version: Tensile_version, target_id: "ON")
                    }
                }
            }
        }
        stage("Packages"){
            when { expression { params.PACKAGES && !params.DISABLE_ALL_STAGES } }
            parallel {
                stage('OpenCL Package') {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildHipClangJobAndReboot(compiler: 'g++', package_build: "true", gpu_arch: "gfx900;gfx906;gfx908")
                    }
                }
                stage("HIP Package /opt/rocm"){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildHipClangJobAndReboot( package_build: "true", prefixpath: '/opt/rocm', gpu_arch: "gfx900;gfx906;gfx908")
                    }
                }
            }
        }
    }
}

