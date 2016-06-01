#!/usr/bin/env python

from setuptools import setup

version = '1.0.0'

required = open('requirements.txt').read().split('\n')

setup(
    name='factorization',
    version=version,
    description='algorithms for scalable matrix factorization',
    author='jwittenbach',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/thunder-project/thunder-factorization',
    packages=['factorization', 'factorization.algorithms'],
    install_requires=required,
    long_description='See ' + 'https://github.com/thunder-project/thunder-factorization',
    license='MIT'
)
