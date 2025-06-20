from setuptools import setup
import io

setup(
    name='promoterai',
    packages=['promoterai'],
    version='1.0rc3',
    description='Predict the impact of promoter variants on gene expression',
    long_description=io.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Illumina/PromoterAI',
    license='PolyForm-Strict-1.0.0',
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.26.2',
        'pandas>=2.1.4',
        'pyfaidx>=0.8.1.1',
        'pybigwig>=0.3.22',
        'tensorflow>=2.13,<2.16'
    ],
    entry_points={'console_scripts': ['promoterai=promoterai.score:main']},
    author='Kishore Jaganathan',
    author_email='kishorejaganathan@gmail.com'
)
