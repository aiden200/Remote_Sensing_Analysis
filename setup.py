from setuptools import setup, find_packages

setup(
    name='Remote_Sensing_Analysis',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add your project dependencies here
    ],
    python_requires='>=3.8',
    author='Aiden Chang',
    author_email='aidenchang@gmail.com',
    description='Library that conducts analysis on Satellite Imagery',
    license='MIT',
    keywords='Machine-Learning Remote-Sensing',
    url='http://github.com/aiden200/Remote_Sensing_Analysis',
)