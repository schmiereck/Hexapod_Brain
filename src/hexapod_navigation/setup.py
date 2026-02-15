from setuptools import setup
import os
from glob import glob

package_name = 'hexapod_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hexapod Team',
    maintainer_email='hexapod@example.com',
    description='Navigation and behavior logic for the Hexapod robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bottle_seeker = hexapod_navigation.bottle_seeker:main',
        ],
    },
)
