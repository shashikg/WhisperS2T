from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="whisper_s2t",
    version="2.0.0-dev-1.8",
    description="An Optimized Speech-to-Text Pipeline for the Whisper Model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    author="Shashi Kant Gupta",
    url="https://github.com/shashikg/WhisperS2T",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements,
    package_data={
        '': ['assets/*'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'whisper-s2t-server=whisper_s2t_server.cli:main',
        ],
    },
)
