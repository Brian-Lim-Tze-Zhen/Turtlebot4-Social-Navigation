from setuptools import find_packages, setup

package_name = 'social_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_viewer = social_perception.camera_viewer:main',
            'yolo_detector = social_perception.yolo_detector:main',
            'move_person_gazebo = social_perception.move_person_gazebo:main',
            'move_person_gazebo2 = social_perception.move_person_gazebo2:main',
            'predicted_person_cloud_node = social_perception.predicted_person_cloud_node:main',
            'human_kf_predictor = social_perception.human_kf_predictor:main',
            'prediction_marker_node = social_perception.prediction_marker_node:main',
            'persistent_map_kf_node = social_perception.persistent_map_kf_node:main',
            'yolo_detector_sync = social_perception.yolo_detector_sync:main',
            'human_kf_predictor_gated = social_perception.human_kf_predictor_gated:main',
        ],
    },
)
