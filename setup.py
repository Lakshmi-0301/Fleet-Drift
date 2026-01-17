"""
Setup file for the anomaly detection package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="vehicle-anomaly-detection",
    version="1.0.0",
    author="Fleet Analytics Team",
    author_email="analytics@fleetmanagement.com",
    description="Anomaly detection system for vehicle trip telemetry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourcompany/vehicle-anomaly-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "detect-anomalies=main:main",
        ],
    },
)