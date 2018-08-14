from setuptools import setup

setup(name='recordlinker',
      version='0.1',
      description='Record linking with blocking using deep learning',
      license='MIT',
      keywords='recordlinker autoencoder blocking record linkage',
      install_requires=[
          'numpy',
          'pandas',
          'tensorflow'
      ],
      test_suite='nose.collector',
      tests_require=['nose']
      )
