import sqlite3
import requests
import os
from os.path import join

import re
from io import BytesIO
import json

from PIL import Image

import pandas as pd
from time import sleep
import math

def roundup(x):
	return int(math.ceil(x/100.0)) * 100

class_path = "picturelikesurvey/static/instagram/"
folders = [f for f in os.listdir(class_path) if os.path.isdir(join(class_path, f))]
subscription_key = "ad7dc4ae03fb4f8ea68ca99109367a35"
assert subscription_key
vision_base_url = "https://eastasia.api.cognitive.microsoft.com/vision/v1.0/"

analyze_url = vision_base_url + "analyze"

file_list = [ f for f in os.listdir("crawler/instagram/") ]
file_list.sort()

df = pd.DataFrame()
cols = ['name', 'domain', 'category', 'background', 'foreground', 'sentence', 'tags', 'caption']

for f in file_list:

	image_path = ""
	for folder in folders:
		files = [f for f in os.listdir(class_path + folder)]
		if f in files:
			image_path = class_path + folder + '/' + f
			break
	
	if image_path == "":
		print("Removed file ", f)
		print("############################################################")
		continue

	print("Path: ", image_path)

	image_data = open(image_path, "rb").read()
	headers = {'Ocp-Apim-Subscription-Key': subscription_key,
				'Content-Type': 'application/octet-stream'}
	params = {'visualFeatures': 'Categories,Description,Color'}
	while True:
		try:
			response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
			response.raise_for_status()
			break
		except:
			print("Response Error\n##################")
			sleep(30)

	analysis = response.json()

	print(json.dumps(analysis, indent=4, sort_keys=True))

	picnum = re.findall(r'\d+', f)
	picnum = int(picnum[0])

	with open(class_path + "data" + str(roundup(picnum)).zfill(4) + '.json') as jsonfile:
		data = json.load(jsonfile)
		
		for pic_explain in data['picture_list']:
			if str(picnum).zfill(4) in re.findall(r'\d+', pic_explain['path']):
				pattern = re.compile("\(\)")
				caption = re.sub(pattern, '', pic_explain['caption'])
				break

		jsonfile.close()


	#folder
	try:
		category_name = analysis['categories'][0]['name']
	except:
		category_name = ""

	try:
		background = analysis['color']['dominantColorBackground']
	except:
		background = ""

	try:
		foreground = analysis['color']['dominantColorForeground']
	except:
		foreground = ""

	try:
		sentence = analysis['description']['captions'][0]['text']
	except:
		sentence = ""

	try:
		tags = '#'.join(analysis['description']['tags'])
	except:
		tags = ""

	#caption
#
#	print(category_name)
#	print(background)
#	print(foreground)
#	print(sentence)
#	print(tags)
#	print(caption)

	temp_df = pd.DataFrame([[f, folder, category_name, background, foreground, sentence, tags, caption]], columns=cols)

	df = df.append(temp_df, ignore_index=True)
	if picnum % 10 == 0:
		df[cols].to_csv("features.csv", sep=',', encoding='utf-8')

df[cols].to_csv("features.csv", sep=',', encoding='utf-8')


