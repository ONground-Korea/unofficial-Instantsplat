import os
import sqlite3
import pickle

from matplotlib import table

def read_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM images")
    images_tuples = c.fetchall()

    c.execute("SELECT * FROM cameras")
    cameras_tuples = c.fetchall()

    c.execute("SELECT * FROM keypoints")
    keypoints_tuples = c.fetchall()

    return cameras_tuples, images_tuples, keypoints_tuples

def modify_db(db_path, data):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM images")
    images_tuples = c.fetchall()
    images_lists = []
    for idx, image in enumerate(images_tuples):
        images_list = list(image)
        images_list[3:] = list(data[idx].values())[0]
        images_lists.append(images_list)

    # update the "images" column
    # swap the order of the columns (images_lists[:][0] and images_lists[:][-1])
    images_lists = [i[1:] + i[:1] for i in images_lists]
    c.executemany("UPDATE images SET name = ?, camera_id = ?, prior_qw = ?, prior_qx = ?, prior_qy = ?, prior_qz = ?, prior_tx = ?, prior_ty = ?, prior_tz = ? WHERE image_id = ?", images_lists)
    # finish the transaction
    conn.commit()
    conn.close()


cams, images, keypoints = read_db("/home/cvlab05/project/sda/colmap/bicycle/database_perturb.db")
qt = pickle.load(open("cam_perturb_qt.pkl", "rb"))
# t = pickle.load(open("cam_perturb_t.pkl", "rb"))
import ipdb; ipdb.set_trace()

modify_db("/home/cvlab05/project/sda/colmap/bicycle/database_perturb.db", qt)

# import sqlite3
# import pandas as pd
# import os

# # Step 1: Read SQLite database and save to txt files
# def db_to_txt(db_path):
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # Get the list of all tables
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     tables = cursor.fetchall()

#     for table_name in tables:
#         table_name = table_name[0]
#         table = pd.read_sql_query("SELECT * from %s" % table_name, conn)
#         table.to_csv(table_name + '.txt', index_label='index')

#     conn.close()
#     return tables

# # Step 2: Modify txt files manually

# # Step 3: Read modified txt files and create a new SQLite database
# def txt_to_db(db_path):
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # Get the list of all txt files
#     tables = ['cameras', 'images']

#     for table_name in tables:
#         table = pd.read_csv(table_name + '.txt', index_col='index')
#         print(table)

#         table.to_sql(table_name, conn, if_exists='replace')

#     conn.close()

# # Convert db to txt
# _ = db_to_txt('/home/cvlab05/project/sda/colmap/bicycle/database_perturb.db')

# # read txt file
# with open('images.txt', 'r') as f:
#     lines = f.readlines()

# import pickle
# qt = pickle.load(open("cam_perturb_qt.pkl", "rb"))

# # modify txt file
# new_lines = []
# for li in lines:
#     li_list = li.split(',')
#     # if li_list[0] is a number
#     if li_list[0] != 'index':
#         li_list[4:] = [str(i) for i in list(qt[int(li_list[0])].values())[0]]
#         new_lines.append(','.join(li_list)+'\n')
#         continue
#     new_lines.append(','.join(li_list))

# with open('images.txt', 'w') as f:
#     for line in new_lines:
#         f.write(line)
# # Convert txt back to db
# txt_to_db('new_database.db')