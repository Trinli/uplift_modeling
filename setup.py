from setuptools import setup, find_namespace_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='uplift_modeling',
      version='0.1',
      description='Models, metrics, and data handling for uplift modeling.',
      url='https://github.com/trinli/uplift_modeling',
      author='Otto Nyberg',
      packages=['uplift_modeling'],
      install_requires=[
          'numpy',
          'numba',
          'scipy',
          'scikit-learn',
          'torch',
          'matplotlib',
          'pytest'
      ],
       extras_require={
           'extras': [
               'rpy2',
               'gpflow',
               'heteroscedastic']},
      zip_safe=False,
      include_package_data=True)
