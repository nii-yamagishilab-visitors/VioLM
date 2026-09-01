from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).parent

setup(
    name="violm-ismir26",
    version="1.0.0",
    description="VioLM expressive violin synthesis",
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.11",
    license="BSD-3-Clause AND Apache-2.0",
    license_files=["LICENSE", "LICENSES/Apache-2.0.txt"],
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    install_requires=[
        "audiocraft==1.3.0",
        "encodec==0.1.1",
        "h5py==3.11.0",
        "librosa==0.11.0",
        "lhotse==1.32.2",
        "loralib==0.1.2",
        "matplotlib==3.8.4",
        "miditok==3.0.6.post1",
        "mido==1.3.3",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "phonemizer==3.3.0",
        "pretty-midi==0.2.10",
        "soundfile==0.12.1",
        "symusic==0.6.0",
        "tensorboard==2.17.1",
        "torchmetrics==0.11.1",
        "tqdm==4.67.1",
        "transformers==4.41.2",
    ],
)
