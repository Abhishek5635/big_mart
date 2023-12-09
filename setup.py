from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e'

def get_requirements(filepath : str)-> List[str]:
    requirement = []
    with open(filepath, ) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n',"") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove('HYPEN_E_DOT')
    return requirements

setup(
    name = "BigMart_Sales_Project",
    version='0.0.1',
    author='Abhishek Wankhade',
    author_email="abhishekwankhade4670@gmail.com",
    install_requires = get_requirements('requirements.txt'),
    packages=find_packages()
)