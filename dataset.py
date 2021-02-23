import re
import os
import glob
import shutil
import sys

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

if __name__ == "__main__":
    LOG_FILE = "Steer_data_5200.txt"
    IMAGES_FOLDER = "Images_5200"

    if not (os.path.isfile(LOG_FILE) and os.path.isdir(IMAGES_FOLDER)):
        sys.exit("Error: Can't Find '{}' or '{}' folder!".format(LOG_FILE, IMAGES_FOLDER))

    with open(LOG_FILE) as f:
        steer_list = f.readlines()

    steer_list = [x.strip() for x in steer_list]
    image_list = glob.glob("{}/*.jpg".format(IMAGES_FOLDER))
    image_list = sorted_nicely(image_list)

    if len(steer_list) != len(image_list):
        sys.exit("Error: Images count must be equal to Steer data!")

    steer_values = set(steer_list)
    for dir in steer_values:
        dir = 'dataset/{}'.format(int(dir) + 3)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    for i in range(len(image_list)):
        name = os.path.basename(image_list[i])
        src = image_list[i]
        dist = 'dataset/{}/{}'.format(int(steer_list[i]) + 3, name)
        shutil.copyfile(src, dist)
        print("Moved {} to {}".format(src, dist))
