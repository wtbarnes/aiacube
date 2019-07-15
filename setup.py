import os
import setuptools
from distutils.core import setup

# create home directory
if not os.path.isdir(os.path.join(os.environ['HOME'], '.hissw')):
    os.mkdir(os.path.join(os.environ['HOME'], '.hissw'))

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='hissw',
    license='MIT',
    version='1.0',
    description='Data containers for analyzing AIA data cubes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Will Barnes',
    author_email='will.t.barnes@gmail.com',
    url='https://gitlab.com/wtbarnes/aiacube',
    packages=['aiacube'],
    classifieers=[
        'Development Status :: 5 - Production/Stable',
        'Intendend Audience :: Scientists',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='solar sun ssw solar-physics',
    project_urls={
        'Source': 'https://gitlab.com/wtbarnes/aiacube',
    },
    install_requires=[
        'astropy',
        'sunpy',
        'numpy',
        'dask',
        'distributed',
        'ndcube',
        'scipy',
        'distributed'
    ],
    python_requires='>=3.6'
)
