from setuptools import setup, find_packages

setup(
    name="clape",  
    version="0.1.1",  
    author="Andrew Liu",
    author_email="andyalbert97@gmail.com",
    description="clape",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YAndrewL/CLAPE",
    packages=find_packages(), 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts':[
            'clape=clape.clape_cli:main'
        ]
    }
)