import setuptools
from distutils.core import setup


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='aiacube',
    license='MIT',
    version='0.0',
    description='Data containers for analyzing AIA data cubes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Will Barnes',
    author_email='will.t.barnes@gmail.com',
    url='https://gitlab.com/wtbarnes/aiacube',
    packages=['aiacube'],
    classifiers=[
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
        #'ndcube',  # version requirements for astropy are wrong
        'scipy',
        'distributed'
    ],
    python_requires='>=3.6'
)
