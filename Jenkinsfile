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
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg MIOTENSILE_VER='${miotensile_version}' --build-arg USE_TARGETID='${target_id}' --build-arg USE_MLIR='${mlir_build}' "

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
/// Target := { gfx908 | Vega20 | Vega10 | Vega* }
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
                stage('Fp16 Hip All Install gfx908') {
                    agent{ label rocmnode("gfx908") }
                    steps{
                        buildHipClangJobAndReboot(setup_flags: Full_test + Fp16_flags, build_env: WORKAROUND_iGemm_936, build_install: "true", gpu_arch: "gfx908")
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

