import os

# Images Data

content_images_file = ['Pedestrians on the highway1.jpeg', 'test2.jpeg', 'nopedstrain1.jpeg', '']

content_images_name = ['Pedestrians on the highway', 'biker and traffic light', 'no pedstrain', 'jaywalkers']

images_path = 'image'

content_images_dict = {name: os.path.join(images_path, filee) for name, filee in zip(content_images_name, content_images_file)}

