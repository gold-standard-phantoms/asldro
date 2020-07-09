import os

from setuptools import setup, find_packages

if os.environ.get("CI_COMMIT_TAG"):
    version = os.environ["CI_COMMIT_TAG"]
else:
    version = "dev{version}".format(version=os.environ.get("CI_JOB_ID", 0))

with open(os.path.join("requirements", "base.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="ASLDRO",
    description="ASL Digital Reference Object",
    version=version,
    author="Gold Standard Phantoms",
    author_email="info@goldstandardphantoms.com",
    license="Commercial",
    url="https://goldstandardphantoms.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    include_package_data=True,
)