import os
import shlex
import shutil
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class BuildArchitectureTests(unittest.TestCase):
    def run_make(self, arch: str | None = None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        for name in ("ARCH", "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "MAKEFILES"):
            environment.pop(name, None)
        command = [
            "make",
            "--always-make",
            "--dry-run",
            "NVCC=nvcc",
            "SRC=csrc/bindings.cu",
            "OUT=build-dry-run.so",
            "PYTHON_INCLUDES=",
            "PYTORCH_INCLUDES=",
            "PYTORCH_LIBDIR=",
            "all",
        ]
        if arch is not None:
            command.append(f"ARCH={arch}")
        return subprocess.run(
            command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True
        )

    def build_flags(self, arch: str | None = None) -> list[str]:
        result = self.run_make(arch)
        self.assertEqual(result.returncode, 0, result.stderr)
        return shlex.split(result.stdout)

    def assert_architectures(self, flags: list[str], architectures: list[str], macro: str) -> None:
        self.assertCountEqual(
            [flags[index + 1] for index, flag in enumerate(flags) if flag == "-gencode"],
            [f"arch=compute_{arch}a,code=sm_{arch}a" for arch in architectures],
        )
        self.assertEqual([flag for flag in flags if flag.startswith("-DKITTENS_SM")], [macro])

    def test_default_architecture(self) -> None:
        self.assert_architectures(self.build_flags(), ["103"], "-DKITTENS_SM103")

    def test_single_architecture(self) -> None:
        for arch in ("100", "103", "107"):
            with self.subTest(arch=arch):
                flags = self.build_flags(f"SM{arch}")
                self.assert_architectures(
                    flags, [arch], f"-DKITTENS_SM{arch}"
                )
                self.assertEqual(
                    [flag for flag in flags if flag.startswith("-DMOK_BUILD_SM")],
                    [f"-DMOK_BUILD_SM{arch}"],
                )

    def test_all_architectures(self) -> None:
        flags = self.build_flags("ALL")
        self.assertCountEqual(
            [flags[index + 1] for index, flag in enumerate(flags) if flag == "-gencode"],
            [f"arch=compute_{arch}a,code=sm_{arch}a" for arch in ("100", "103", "107")],
        )
        self.assertFalse(any(flag.startswith("-DKITTENS_SM") for flag in flags))
        self.assertIn("-DMOK_ARCH_ALL", flags)
        self.assertCountEqual(
            [flag for flag in flags if flag.startswith("-DMOK_BUILD_SM")],
            [f"-DMOK_BUILD_SM{arch}" for arch in ("100", "103", "107")],
        )
        self.assertEqual(flags[flags.index("-include") + 1], "csrc/architecture.cuh")
        self.assertIn("--threads=2", flags)

    def test_unsupported_architecture(self) -> None:
        result = self.run_make("SM90")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported ARCH 'SM90'", result.stderr)
        for supported in ("SM100", "SM103", "SM107", "ALL"):
            self.assertIn(supported, result.stderr)

    @unittest.skipUnless(shutil.which("c++"), "a C++ preprocessor is required")
    def test_all_selects_one_architecture_per_compilation(self) -> None:
        for device_arch, expected in ((None, "100"), (1000, "100"), (1030, "103"), (1070, "107")):
            with self.subTest(device_arch=device_arch):
                command = ["c++", "-E", "-dM", "-x", "c++", "-DMOK_ARCH_ALL"]
                if device_arch is not None:
                    command.append(f"-D__CUDA_ARCH__={device_arch}")
                command.append("csrc/architecture.cuh")
                result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    [line.strip() for line in result.stdout.splitlines() if line.startswith("#define KITTENS_SM")],
                    [f"#define KITTENS_SM{expected}"],
                )

    @unittest.skipUnless(shutil.which("c++"), "a C++ preprocessor is required")
    def test_all_rejects_unsupported_device_compilation(self) -> None:
        result = subprocess.run(
            ["c++", "-E", "-x", "c++", "-DMOK_ARCH_ALL", "-D__CUDA_ARCH__=900", "csrc/architecture.cuh"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MoK supports only SM100, SM103 and SM107", result.stderr)


if __name__ == "__main__":
    unittest.main()
