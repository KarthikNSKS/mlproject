from setuptools import find_packages,setup
from typing import List

HYPEN_E="-e ."

def get_requirements(file_path:str)->List[str]:
    
    requirements=[]
    with open(file_path)  as file_obj:
        requirements = file_obj.readlines()
        requirements = [items.replace("\n","") for items in requirements]

        if HYPEN_E in requirements:
            requirements.remove(HYPEN_E)

    return requirements

setup(
name ='mlproject',
version = '0.0.1',
author='Karthik',
author_email='karthikns.inbox@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)