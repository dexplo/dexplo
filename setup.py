from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='dexplo',
      version='0.0.1',
      description='A library for data exploration comparible to pandas. '
                  'No Series, No hierarchical indexing, only one indexer [ ]',
      long_description=readme(),
      url='https://github.com/dexplo/dexplo',
      author='Ted Petrou',
      author_email='petrou.theodore@gmail.com',
      license='BSD 3-clause',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6'],
      keywords='data analysis exploration aggregation pandas numpy',
      packages=find_packages(exclude=['docs', 'stubs']),
      install_requires=['numpy'],
      python_requires='>=3.6',
      include_dirs=[np.get_include()],
      ext_modules=cythonize('dexplo/_libs/*.pyx'))
