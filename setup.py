from setuptools import setup, find_packages

setup(
  name = 'kronecker-attention-pytorch',
  packages = find_packages(),
  version = '0.0.3',
  license='MIT',
  description = 'Kronecker Attention - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/kronecker-attention-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism'
  ],
  install_requires=[
    'torch',
    'einops'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)