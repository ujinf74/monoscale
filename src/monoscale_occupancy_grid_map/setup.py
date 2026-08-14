from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'monoscale_occupancy_grid_map'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ujin Kwon',
    maintainer_email='ujin@tukorea.ac.kr',
    description='An occupancy grid accumulated from the labelled ground points the odometry publishes.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'monoscale_occupancy_grid_map = monoscale_occupancy_grid_map.occupancy_grid_map:main',
        ],
    },
)
