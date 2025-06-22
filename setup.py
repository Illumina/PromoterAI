from setuptools import setup
import io

setup(
    name='promoterai',
    packages=['promoterai'],
    version='1.0rc6',
    description='Predict the impact of promoter variants on gene expression',
    long_description=io.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Illumina/PromoterAI',
    license='PolyForm Strict License 1.0.0',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
        'pandas>=2.0',
        'pyfaidx>=0.8',
        'pybigwig>=0.3',
        'tensorflow>=2.13,<2.16',
        'keras>=2.13,<3.0'
    ],
    entry_points={'console_scripts': ['promoterai=promoterai.score:main']},
    author='Kishore Jaganathan',
    author_email='kishorejaganathan@gmail.com'
)
