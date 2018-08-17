from setuptools import setup

setup(name='recordlinker',
      version='0.1',
      description='Record linking with blocking using deep learning',
      license='MIT',
      keywords='recordlinker autoencoder blocking record linkage',
      install_requires=[
          'numpy>=1.0',
          'pandas>=0.22',
          'tensorflow>=1.4',
          'Keras>=2.0',
          'pyjarowinkler>=1.0'
      ],
      test_suite='nose.collector',
      tests_require=['nose']
      )
