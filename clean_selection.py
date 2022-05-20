import os
import random
import shutil

images = os.listdir("clean/")
random_images = random.choices(images,k = 1308)
for imageName in random_images:
    shutil.copyfile("clean/" + imageName,"clean_selected/" + imageName)