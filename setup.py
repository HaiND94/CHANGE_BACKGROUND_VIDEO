import pathlib

from setuptools import find_packages, setup

import background_process

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    requireds = f.read().splitlines()

setup(
    install_requires=requireds,
    name = 'background_process',
    description = 'A module to process background remove or change',
    version = 0.1,
    author = "HaiND",
    author_email="haind.ee.094@hotmail.com",
    url = 'https://git.vfastsoft.com/VFAST/Background_removal.git',
    license = 'GPLv3',
    python_requires=">=3.6, <4",
    packages = find_packages(where="background_process"),
    entry_points  = {
            'console_scripts': [
                'background = background_process.__main__:background_remove'
            ]
        },
)