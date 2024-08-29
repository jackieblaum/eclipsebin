from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eclipsebin",
    version="0.1.1",
    author="Jackie Blaum",
    author_email="jackie.blaum@example.com",
    description="A specialized binning scheme for eclipsing binary star light curves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jackieblaum/eclipsebin",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
    ],
)
