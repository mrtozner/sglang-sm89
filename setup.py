from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXT_SOURCES = [
    "csrc/torch_bindings.cpp",
    "csrc/coupled_grammar_cpu.cpp",
    "csrc/sparse_vocab_proj_sm89.cu",
    "csrc/wy_gdn_sm89.cu",
]

CXX_OPTS = ["-O3", "-std=c++20"]
NVCC_OPTS = [
    "-O3",
    "-std=c++20",
    "-gencode=arch=compute_89,code=sm_89",
    "--ptxas-options=-v",
    "-Xcudafe",
    "--diag_suppress=3189",
]

setup(
    name="sglang-sm89",
    version="0.3.2",
    description="Ada Lovelace (sm_89) SGLang speculative & constrained decoding acceleration",
    license="Apache-2.0",
    packages=find_packages(include=["sglang_sm89*"]),
    entry_points={
        "sglang.plugins": ["sglang_sm89 = sglang_sm89.plugin:register"],
        "sglang.srt.plugins": ["sglang_sm89 = sglang_sm89.plugin:register"],
    },
    ext_modules=[
        CUDAExtension(
            name="sglang_sm89._C",
            sources=EXT_SOURCES,
            extra_compile_args={"cxx": CXX_OPTS, "nvcc": NVCC_OPTS},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
