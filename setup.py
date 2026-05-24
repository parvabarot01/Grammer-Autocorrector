from setuptools import find_packages, setup


setup(
    name="grammar-autocorrector",
    version="0.1.0",
    description="Production-grade NLP grammar autocorrector system scaffold.",
    author="Parva Barot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
)
