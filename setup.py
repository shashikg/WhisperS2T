import os
import pkg_resources
from setuptools import setup

requirements_file_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
requirements = [str(r) for r in pkg_resources.parse_requirements(open(requirements_file_path))]

setup(
    name="whisper_s2t",
    version="1.0.0",
    description="An Optimized Speech-to-Text Pipeline for the Whisper Model.",
    readme="README.md",
    python_requires=">=3.8",
    author="Shashi Kant Gupta",
    url="https://github.com/shashikg/WhisperS2T",
    license="MIT",
    packages=['whisper_s2t'],
    install_requires=requirements,
    include_package_data=True,
)