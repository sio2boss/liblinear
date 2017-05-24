from setuptools import setup
setup(
  name = 'liblinear',
  packages = ['liblinear'],
  version = '2.11.2',
  description = 'Python bindings for liblinear',
  author = 'sio2boss',
  author_email = 'sio2boss@hotmail.com',
  url = 'https://github.com/sio2boss/liblinear',
  download_url = 'https://github.com/sio2boss/liblinear/tarball/2.11.2',
  keywords = ['liblinear'],
  classifiers = [],
  entry_points = {
    "console_scripts": ['liblinear = liblinear:main']
  },
  install_requires=[
    "scipy"
  ],
)
