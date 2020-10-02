import os

from setuptools import setup, find_packages

if os.environ.get("CI_COMMIT_TAG"):
    version = os.environ["CI_COMMIT_TAG"]
else:
    version = "dev{version}".format(version=os.environ.get("CI_JOB_ID", 0))

with open(os.path.join("requirements", "base.txt")) as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="asldro",
    version=version,
    author="Gold Standard Phantoms",
    author_email="info@goldstandardphantoms.com",
    description="ASL Digital Reference Object",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/gold-standard-phantoms/asldro",
    project_urls={
        "Documentation": "https://asldro.readthedocs.io/",
        "Code": "https://github.com/gold-standard-phantoms/asldro",
        "Issue tracker": "https://github.com/gold-standard-phantoms/asldro/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=True,
    entry_points={"console_scripts": ["asldro=asldro.cli:main"]},
)
