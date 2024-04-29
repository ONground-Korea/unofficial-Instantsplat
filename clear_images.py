import os
import sqlite3

new_lines = []
with open("/home/cvlab05/project/sda/colmap/bicycle/sparse/0/images.txt",'r') as f:
    lines = f.readlines()

import ipdb; ipdb.set_trace()

for li in lines:
    image_name = li.split(" ")[-1]
    if ".JPG" in image_name:
        new_lines.append(image_name)
    if "frame" in image_name and "_1_" not in image_name:
        frame_id = str(int(image_name.split("_")[1][:5]))
        new_li = " ".join([frame_id] + li.split(" ")[1:])

        new_lines.append(new_li)
        # new_lines.append("\n")

# new_lines.sort(key=lambda x: int(x.split(" ")[0]))

with open("images.txt",'w') as f:
    for l in new_lines:
        f.write(l)
        f.write("\n")


# def read_db(db_path):
#     conn = sqlite3.connect(db_path)
#     c = conn.cursor()
#     c.execute("SELECT * FROM images")
#     images_tuples = c.fetchall()

#     c.execute("SELECT * FROM cameras")
#     cameras_tuples = c.fetchall()

#     return cameras_tuples, images_tuples


# cams, images = read_db("/home/ubuntu/data/nerfbusters-dataset/pikachu_train/colmap/sparse/0/database.db")

# breakpoint()