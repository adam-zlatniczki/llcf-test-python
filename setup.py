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
        'numpy',
        'scipy',
        'pandas',
        'matplotlib'
    ]
)
