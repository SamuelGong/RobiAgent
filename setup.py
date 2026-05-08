from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="robiagent",
    version="0.1",
    description="'RobiAgent' is your command-line agent for physical intelligence.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SamuelGong/RobiAgent",
    license="Apache 2.0",
    packages=find_packages(exclude=("log", "examples", "rdb")),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ra=robiagent.cli:main",  # Maps 'ra' command to quick_start.py:main()
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="Embodied Intelligence, Physical AI, Robotics",
    python_requires='>=3.12',
)
