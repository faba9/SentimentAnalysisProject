from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import io
import urllib, base64
from PIL import Image
import pandas as pd
import numpy as np
from modly import *

df = pd.read_csv('/home/me/jupyter-workspace/Amazon_Unlocked_Mobile.csv')
df.dropna(inplace=True)
products_names = [i for i in df['Product Name'].tolist()]
model = load_model('/home/me/jupyter-workspace/my_model.h5')
max_features = 20000
maxlen = 100
# ---------------------------------

def user_order(order):
	result = ''

	if order in products_names:

		# take the input and get the reviews of that input / done
		# convert each block of reviews to array
		# make predictions on this array 
		# get the result of each comment if it positive or negative or natural
		# calculate the percetage of each type

		# list_of_rows = df.loc[df['Product Name'] == order , 'Reviews']
		list_of_reviews = [i for i in df.loc[df['Product Name'] == order, 'Reviews']]

		tokenizer = Tokenizer(nb_words = max_features, split=' ')
		tokenizer.fit_on_texts(list_of_reviews)
		X = tokenizer.texts_to_sequences(list_of_reviews)
		X = pad_sequences(X, maxlen = maxlen)

		negatives = 0
		positives = 0
		naturals = 0

		for i in range(len(X)):
		    v = np.array([X[i]])
		    m = model.predict(v)

		    prediction_list = []

		    for o in m:
		        for x in o:
		            prediction_list.append(x)

		    natural, negative, positive = prediction_list
		    if max(prediction_list) == natural:
		        naturals += 1
		    elif max(prediction_list) == negative:
		        negatives += 1
		    elif max(prediction_list) == positive:
		        positives += 1

		 # calculate result

		all_numbers = naturals + negatives + positives

		positive_percentage = int(round((positives/all_numbers) * 100, 1))
		negative_percentage = int(round((negatives/all_numbers) * 100, 1))
		natural_percentage = int(round((naturals/all_numbers) * 100, 1))
		
		res = [positive_percentage, negative_percentage, natural_percentage]
		result = res
		
	else:

		result = 'Product not found!'

	return result

  
def get_reviews(product):

	negatives = []
	positives = []
	list_of_reviews = [i for i in df.loc[df['Product Name'] == product, 'Reviews']]
	list_of_reviews = list(set(list_of_reviews))
	for i in range(len(list_of_reviews)):
		prediction = create_predict(list_of_reviews[i])

		if prediction == 0:
			negatives.append(list_of_reviews[i])
		elif prediction == 1:
			positives.append(list_of_reviews[i])
		else:
			nothing.append(list_of_reviews[i])

	p = []
	n = []
    
	if len(positives) > 3:
		for i in range(3):
			p.append(positives[i])
	else:
		for i in positives:
			p.append(i)
    
	if len(negatives) > 3:
		for i in range(3):
			n.append(negatives[i])
	else:
	    for i in negatives:
	        n.append(i)
	        # old : n.append(negatives[i])
    

	all_reviews = [p,n]
	return all_reviews


def get_pos_rev(product):
	reviews = get_reviews(product)

	pos = reviews[0]
	# pos1, pos2, pos3 = pos
	return pos

def get_neg_rev(product):
	reviews = get_reviews(product)

	neg = reviews[1]
	# pos1, pos2, pos3 = pos
	return neg

def gen_cloud(product):

	list_of_reviews = [i for i in df.loc[df['Product Name'] == product, 'Reviews']]

	letters_only = re.sub("[^a-zA-Z]", " ",str(list_of_reviews))

	stopwords = set(STOPWORDS)
	text = letters_only

	mask = np.array(Image.open("/home/me/project/static/mask.png"))

	image_colors = ImageColorGenerator(mask)
	wc = WordCloud(background_color="white", max_words=10000, mask=mask,
	               stopwords=stopwords)
	# Generate a wordcloud
	wc.generate(text)


	# show
	plt.figure(figsize=[20,20])
	plt.imshow(wc, interpolation='bilinear')
	plt.axis("off")

	image = io.BytesIO()
	plt.savefig(image, format='png')
	image.seek(0)  # rewind the data
	string = base64.b64encode(image.read())

	image_64 = 'data:image/png;base64,' + urllib.parse.quote(string)
	
	return image_64

# this is my code 
# user = 'LG Extravert 2, Verizon (Blue) - Retail Packaging'

def get_brand_name(product):
	
	brand = df.loc[df['Product Name'] == product, 'Brand Name']

	brand = list(brand)
	if len(brand) >=1:
		brand = brand[0]
	else:
		brand = 'Unknown'
	return brand

def get_price_brand(product):
	brand = get_brand_name(product)
	try:
		how_many_products = df.loc[df['Brand Name'] == brand, 'Product Name']
		how_many_products = list(set(how_many_products))
		
		new_prices = []
		for product_test in how_many_products:
		    price = df.loc[df['Product Name'] == product_test, 'Price']
		    price = list(set(price))
		    price = price[0]
		    new_prices.append(price)

		hundred = 0
		five_hundred = 0
		thousend = 0
		more_thousend = 0

		for p in new_prices:
		    
			if p <= 100:
				hundred += 1
			elif p <= 500:
				five_hundred += 1
			elif p <= 1000:
				thousend += 1
			else :
				more_thousend += 1

		all_categories = hundred + five_hundred + thousend + more_thousend

		per_hundred = round((hundred/ all_categories) *100, 1)
		per_five_hundred = round((five_hundred/ all_categories) *100, 1)
		per_thousend= round((thousend/ all_categories) *100, 1)
		per_more_thousend= round((more_thousend/ all_categories) *100, 1)
		
		res = [per_hundred, per_five_hundred,
		 per_thousend, per_more_thousend]
	except:
		res = [0, 0, 0, 0]
	
	return res

def get_brand_reviews(product):
	brand = get_brand_name(product)

	natural = 0
	negative = 0
	positive = 0

	df.dropna()
	df2 = df.loc[df['Brand Name'] == brand, ['Rating', 'Reviews']]

	pos = df2.loc[df['Rating'] > 3]
	neg = df2.loc[df['Rating'] < 3]
	nat = df2.loc[df['Rating'] == 3 ]

	total = pos.shape[0] + neg.shape[0]+ nat.shape[0]
	pos_per = round((pos.shape[0]/total )*100 ,1)
	neg_per = round((neg.shape[0]/total )*100 ,1)
	nat_per = round((nat.shape[0]/total )*100 ,1)

	all_total = [pos_per, neg_per, nat_per]
	return all_total

# print(user_order(user))