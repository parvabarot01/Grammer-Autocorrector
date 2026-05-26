from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="grammar-autocorrector",
    version="1.0.0",
    description="Production-grade NLP grammar autocorrector with T5, BERT, and RAG.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Parva Barot",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
)
