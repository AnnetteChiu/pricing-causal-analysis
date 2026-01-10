from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pricing-causal-analysis",
    version="1.0.0",
    author="Annette Chiu",
    author_email="",
    description="因果推論在定價策略上的應用 - Causal Inference for Pricing Strategy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnnetteChiu/pricing-causal-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="causal inference, pricing strategy, econometrics, machine learning, A/B testing",
    project_urls={
        "Bug Reports": "https://github.com/AnnetteChiu/pricing-causal-analysis/issues",
        "Source": "https://github.com/AnnetteChiu/pricing-causal-analysis",
        "Documentation": "https://github.com/AnnetteChiu/pricing-causal-analysis/blob/main/使用指南.md",
    },
)