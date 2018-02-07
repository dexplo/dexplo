from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import glob


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

if USE_CYTHON:
    extensions = cythonize('dexplo/_libs/*.pyx')
else:
    extensions = []
    for fn in glob.glob('dexplo/_libs/*.c'):
        extensions.append(Extension(fn[:-2].replace(r'/', '.'), [fn]))


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='dexplo',
      cmdclass = {'build_ext': CustomBuildExtCommand},
      version='0.0.7',
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
      # include_dirs=[np.get_include()],
      ext_modules=extensions)
