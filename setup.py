#!/usr/bin/env python

from distutils.core import setup

setup(
    name='llcf-test',
    version='1.0',
    description='Statistical hypothesis test for the existence of a locally Lipschitz continuous function',
    author='Adam Zlatniczki',
    author_email='adam.zlatniczki@cs.bme.hu',
    packages=['llcf_test'],
    license="MIT",
    install_requires=[
        'numpy>=1.19.2',
        'scipy>=1.5.2',
        'pandas>=1.1.3',
        'matplotlib>=3.3.2'
    ]
)
