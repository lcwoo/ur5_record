#!/usr/bin/env python3
"""Setup script for RFM package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ur5",
    version="0.1.0",
    author="lcw",
    description="UR5 teleop, data collection, and camera stack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lcwoo/ur5_lerobot",
    packages=find_packages(exclude=["tests", "thirdparty", "gello_software", "lerobot", "octo"]),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            # Keep legacy names for compatibility; add UR5-prefixed aliases
            "rfm-run-policy=ur5.policies.runner:main",
            "rfm-ur5-bridge=ur5.robots.ur5_bridge:main",
            "ur5-run-policy=ur5.policies.runner:main",
            "ur5-bridge=ur5.robots.ur5_bridge:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
