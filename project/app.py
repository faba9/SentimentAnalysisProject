from flask import Flask, render_template, request, jsonify
from sen import *

app = Flask('amazon reviews', static_folder='static',)


@app.route('/', methods=['POST', 'GET'])
def index():
	return render_template('index2.html')

@app.route('/analyze', methods=['POST'])
def analyze():
	productName = ''
	productName = request.form.get('productName', '')
	cls = user_order(productName)

	if type(cls) == list:
		pos = get_pos_rev(productName)
		neg = get_neg_rev(productName)
		brand = get_brand_name(productName)
		
		cloud = gen_cloud(productName)
		price_avr = get_price_brand(productName)
		reviews_brand = get_brand_reviews(productName)

		return jsonify({
			'data': cls,
		 	'status': True,
		 	'pos':pos,
		 	'neg':neg,
		 	'cloud': cloud,
		 	'brand':brand,
		 	'price_avr':price_avr,
		 	'reviews_brand':reviews_brand
		})
	else:
		return jsonify({
			'data': None,
			'status': False,
			'cloud': False
		})


@app.route('/names')
def get_all_names():
	return jsonify({'names': list(set(products_names))})
if __name__ == '__main__':
	app.run(debug=True)
