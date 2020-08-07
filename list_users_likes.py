import sqlite3
import requests
import os
from os.path import join

import re
from io import BytesIO
import json

from PIL import Image

import pandas as pd


conn = sqlite3.connect("picturelikesurvey/db.sqlite3")

cur = conn.cursor()
cur.execute("select * from survey_survey")

rows = cur.fetchall()

conn.close()

apostrophe = re.compile("\'")
parentheses_l = re.compile('\[')
parentheses_r = re.compile('\]')

row = rows[0]
discard_row = []

class_path = "picturelikesurvey/static/instagram/"
folders = [f for f in os.listdir(class_path) if os.path.isdir(join(class_path, f))]
subscription_key = "12605cf4d857400fbc796469c5af7498"
assert subscription_key
vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

analyze_url = vision_base_url + "analyze"

df = pd.DataFrame()

for row in rows:
	id_code = row[5]
	raw_select_data = row[3]
	raw_flags_data = row[4]

	select_data = []
	splited_raw_data = re.split(', ', raw_select_data)
	for raw_content in splited_raw_data:
		content = re.sub(apostrophe, '', raw_content)
		content = re.sub(parentheses_l, '', content)
		content = re.sub(parentheses_r, '', content)
		select_data.append(content)

#print(select_data)

	flag_data = raw_flags_data.split(" ")
#print(flag_data)

	if len(flag_data) == 21 or len(str(id_code)) != 7:
		discard_row.append(id_code)
		continue


	idx = 0
	selected_picture = []
	for data in flag_data:
		if data == "1":
			selected_picture.append(select_data[idx])
		idx += 1

	if len(selected_picture) < 10:
		discard_row.append(id_code)
		continue

#print(selected_picture)
	data = {
		'id_code': [id_code] * len(selected_picture),
		'picture': selected_picture
	}

	tmp_df = pd.DataFrame(data=data)
	df = df.append(tmp_df, ignore_index=True)
	print(df)

df.to_csv("user_selection.csv", sep=',', encoding='utf-8')


with open("discards.txt", 'w') as discardsfp:
	for id_code in discard_row:
		discardsfp.write(str(id_code) + '\n')


