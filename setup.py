"""setup.py — pip-installable package for yt-shorts-factory v3.0."""

from pathlib import Path
from setuptools import setup, find_packages

# Read requirements
req_file = Path(__file__).parent / "requirements.txt"
requirements = []
if req_file.exists():
    requirements = [
        line.strip()
        for line in req_file.read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("Optional")
    ]

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="yt-shorts-factory",
    version="4.0.0",
    description="Automated video intelligence pipeline for YouTube Shorts, TikTok, Instagram Reels, and more — by Abrar Hussain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abrar Hussain",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*", "output*"]),
    install_requires=requirements,
    extras_require={
        "fast-whisper": ["faster-whisper>=0.10.0", "ctranslate2>=3.22.0"],
        "face-tracking": ["mediapipe>=0.10.0", "opencv-python>=4.8.0"],
        "visualization": ["matplotlib>=3.8.0"],
        "all": [
            "faster-whisper>=0.10.0",
            "ctranslate2>=3.22.0",
            "mediapipe>=0.10.0",
            "opencv-python>=4.8.0",
            "matplotlib>=3.8.0",
            "tqdm>=4.66.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yt-shorts=main:cli",
            "yt-shorts-factory=main:cli",
        ],
    },
    package_data={
        "": ["*.png", "*.jpg", "*.ttf", "*.otf", "*.json"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
