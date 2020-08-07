import requests
import os
from os.path import join

import re
from io import BytesIO
import json

picture_path = "pictures"
pictures = [f for f in os.listdir(picture_path) if os.path.isfile(join(picture_path, f))]
subscription_key = "9504a746f6e1451f88f7e4dcedb92146"
assert subscription_key
vision_base_url = "https://eastasia.api.cognitive.microsoft.com/vision/v2.0/"

analyze_url = vision_base_url + "analyze"

for picture in pictures:
	image_path = "/home/ftatp/Documents/Studies/personality_consumer_types/pictures/" + picture
	image_data = open(image_path, "rb").read()
	headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
				  'Content-Type': 'application/octet-stream'}
	params     = {'visualFeatures': 'Categories,Description,Color'}
	response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
	response.raise_for_status()

# The 'analysis' object contains various fields that describe the image. The most
# relevant caption for the image is obtained from the 'description' property.
	analysis = response.json()

	json_data = json.dumps(analysis, indent=4, sort_keys=True)
	print(json_data)

	with open("pictures/" + picture.split('.')[0] + ".json", 'w') as outfile:
		json.dump(json_data, outfile)
	
