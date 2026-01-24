from setuptools import setup
import os
from glob import glob

package_name = 'kinect_yellow_ball_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, package_name + '.nodes'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='ROS2 package for Kinect-based yellow ball detection and mapping',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'driver_node = kinect_yellow_ball_detector.nodes.driver_node:main',
            'visualization_node = kinect_yellow_ball_detector.nodes.visualization_node:main',
        ],
    },
)
