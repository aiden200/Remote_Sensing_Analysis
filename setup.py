from setuptools import setup, find_packages

setup(
    name='Remote_Sensing_Analysis',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pillow==10.3.0',
        'ultralytics==8.2.2',
        'transformers==4.40.0',
        'sentence_transformers==2.7.0',
        'sentencepiece==0.2.0',
        'timm==0.9.16',
    ],
    python_requires='>=3.9',
    author='Aiden Chang',
    author_email='aidenchang@gmail.com',
    description='Library that conducts analysis on Satellite Imagery',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', 
    license='MIT',
    keywords='Machine-Learning Remote-Sensing Object-detection',
    url='http://github.com/aiden200/Remote_Sensing_Analysis',
)