from setuptools import find_packages,  setup
from typing import List

constant = '-e .'

def get_requirements(file_path:str)-> List[str]:
    """
    This funciton will return the required pacakages
    """
    require = []
    with open(file_path) as file_obj:
        require = file_obj.readlines()
        require = [req.replace('/n','') for req in require]

    if constant in require:
        require.remove(constant)

    return require


setup(
    name='mlproject',
    version='0.0.1',
    author='Nirmal',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)