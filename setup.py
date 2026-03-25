# coding=utf-8
import os
import sys
from pathlib import Path
from subprocess import DEVNULL, PIPE, run

from setuptools import find_packages, setup

project_root = Path(__file__).parent

# NOTE: This setup.py is intentionally modeled after MIDI-VALLE's setup.py so
# you can run `pip install -e .` in this reconstructed repo.

MAJOR_VERSION = 1
MINOR_VERSION = 0
PATCH_VERSION = 0
IS_DEV_VERSION = True  # False = public release, True = otherwise

if sys.version_info < (3,):
    print("Python 2 has reached end-of-life and is no longer supported.")
    sys.exit(-1)

if sys.version_info < (3, 7):
    print("Python 3.6 has reached end-of-life and is no longer supported.")
    sys.exit(-1)


def discover_valle_version() -> str:
    """
    Scans Valle source code to determine the current version.
    When development version is detected, it queries git for the commit hash
    and appends it as a local version identifier.
    """

    version = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"
    if not IS_DEV_VERSION:
        return version

    version = version + ".dev"

    try:
        git_commit = (
            run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                stdout=PIPE,
                stderr=DEVNULL,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
            len(
                run(
                    ["git", "diff", "--shortstat"],
                    check=True,
                    stdout=PIPE,
                    stderr=DEVNULL,
                )
                .stdout.decode()
                .rstrip("\n")
                .strip()
            )
            > 0
        )
        git_commit = git_commit + (".dirty" if dirty_commit else ".clean")
        source_version = f"+git.{git_commit}"
    except Exception:
        source_version = ".unknownsource"

    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#local-version-identifiers
    version = version + source_version
    return version


def mark_valle_version(version: str) -> None:
    # Write it into `valle/version.py` so `import valle` has a consistent version.
    (project_root / "valle" / "version.py").write_text(
        f'__version__ = "{version}"'
    )


VALLE_VERSION = discover_valle_version()
mark_valle_version(VALLE_VERSION)


install_requires = [
    # Matches MIDI-VALLE's minimal runtime requirements.
    "encodec",
    "phonemizer",
    # `valle.models.visualizer` imports matplotlib at module import time.
    "matplotlib",
]

docs_require = []
docs_req_path = project_root / "docs" / "requirements.txt"
if docs_req_path.is_file():
    docs_require = docs_req_path.read_text(encoding="utf-8").splitlines()

tests_require = []
workflow_requires = [""]
dev_requires = sorted(
    docs_require
    + tests_require
    + workflow_requires
    + ["jupyterlab", "matplotlib"]
)
all_requires = sorted(dev_requires)

if os.environ.get("READTHEDOCS", False):
    # When building documentation, omit torchaudio installation and mock it instead.
    install_requires = [
        req
        for req in install_requires
        if not any(req.startswith(dep) for dep in ["torchaudio", "SoundFile"])
    ]


setup(
    name="valle",
    version=VALLE_VERSION,
    python_requires=">=3.7.0",
    description="Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers",
    author="The Valle Development Team",
    author_email="lifeiteng0422@163.com",
    long_description=(project_root / "README.md").read_text(
        encoding="utf-8"
    ),
    long_description_content_type="text/markdown",
    license="Apache-2.0 License",
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    entry_points={},
    install_requires=install_requires,
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "dev": dev_requires,
        "all": all_requires,
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)

