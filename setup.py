from setuptools import setup, find_packages

setup(
    name='systemflow',
    version='0.1.0',
    description='Modeling dependencies and effects in scientific data processing systems',
    author='Wilkie Olin-Ammentorp',
    author_email='wolinammentorp@anl.gov',
    packages=find_packages(include=['systemflow', 'systemflow.*']),
    # install_requires=[],  # Leave empty if conda handles dependencies
)