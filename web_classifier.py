"""
-------------------------
Web Page Classifier
# Author: Roy Ceyleon
#-------------------------
"""
#-------------------------------------------------------
# IMPORTS
import string
import os
from random import randint
from bs4 import BeautifulSoup as BS
from numpy import log, reshape, dot, zeros, array as npArray, unique
from cloudscraper import create_scraper
from functools import partial
from skll.metrics import kappa as kp
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, matthews_corrcoef, confusion_matrix
from sklearn.model_selection._split import train_test_split as dataset_split
from sklearn.feature_extraction.text import CountVectorizer as getVectorizer
from urllib.parse import urlparse
from urllib import request
from html.parser import HTMLParser as htmlparser
import pickle
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
#-------------------------------------------------------
# SETTINGS 
TRAIN_SET_SIZE = 0.7			# Train Set Size (Default: 70%)
TOPICS_FILE_NAME = 'topics.txt'	# Topic List
DATASET_DIR = 'dataset'			# Data Directory
SESSION_DATA = 'data.dat'	# Session Data File
URL_LIST = 'url_list.txt'	# File containing a list of URLs for user prediction testing
MIN_CHAR_IN_DOC = 80			# Minimum number of characters per document in dataset
#-------------------------------------------------------
# !DO NOT MODIFY!
BASEDIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASEDIR, DATASET_DIR)
SEPARATOR = "="*45
MODEL = None
LINKS = []
VALID = "-_.() %s%s" % (string.ascii_letters, string.digits)
def warn(*args, **kwargs):
	pass
warnings.warn = warn
# !DO NOT MODIFY!
#-------------------------------------------------------

class model(object):
	"""
	Classification model
	"""
	def __init__(self, topics, **kwargs):
		self.topics = topics
		self.vectorizer = getVectorizer(**kwargs)
		self.collection = None
		self.minTopicProb = None

	def predict(self, items):
		"""
		Determine prediction of a given list of documents
		"""
		# Get total number of items to predict
		item_count = len(items)
		
		# Retrieve topics
		topics = self.topics
		
		# Vectorizer to convert a collection of text documents to a matrix of token counts
		v = self.vectorizer
		
		# Initialize analyzer for preprocessing, tokenization and n-grams generation
		analyze = v.build_analyzer()
		
		# Extract token counts out of raw text documents using the vocabulary
		# fitted with fit or the one provided to the constructor.
		data = v.transform(items).toarray()
		
		# Retrieve the word count of each document
		d_sum = data.sum(axis=1)
		ic = (item_count, 1)
		word_count = reshape(d_sum, ic)
		
		# Generate list probabilities for the collection of words
		collection_p = dot(self.collection, data.T)
		
		# Parse vocabulary and to balance probabilities
		total_words = zeros((item_count, 1))
		for i, item in enumerate(items):
			total_words[i, :] = len(analyze(item))
		
		# Filter total word count
		total_words -= word_count
		collection_p = collection_p + (total_words.T * self.minTopicProb)
		
		t_items = topics.__getitem__
		c_max = collection_p.argmax(axis=0)
		return list(map(t_items, c_max))

	def train(self, dataset, labels):
		"""
		Train model using a dataset and the respective labels for each item in the dataset
		"""
		# Get length of dataset
		data_count = len(dataset)
		
		# Retrieve topics
		topics = self.topics
		
		# Get topic count
		topic_count = len(topics)
		labels = npArray(list(map(topics.index, labels)))
		
		# Learn the vocabulary dictionary and return document-term matrix
		data = self.vectorizer.fit_transform(dataset).toarray()

		# Initialize combo matrix
		combo_matrix = zeros((topic_count, data_count))
		for i, _ in enumerate(topics):
			combo_matrix[i, (labels == i)] = 1

		# Combine all words from one class using combo matrix
		data = dot(combo_matrix, data)
		word_count = data.shape[1]

		# Compute logarithmic probabilities using word counts
		# And save the collection of words and their respective counts and probabilities
		d_sum = (data != 0).sum(axis=1)
		tc = (topic_count, 1)
		topic_word_count = reshape(d_sum,tc)
		words = topic_word_count + word_count
		self.collection = log((data+1)/words)
		self.minTopicProb = log(1/words)

def displayConfusionMatrix():
	"""
	Displays the confusion matrix for the assessment of the model using the test set
	"""
	global test_docs, test_classes
	labels = unique(test_classes)
	predicted_classes = MODEL.predict(test_docs)
	confMat = confusion_matrix(predicted_classes, test_classes)
	df = pd.DataFrame(confMat, index=labels, columns=labels)
	seaborn.set(font_scale=0.6)
	seaborn.heatmap(df, annot=True, fmt='g', annot_kws={"size": 10})
	plt.ion()
	
	plt.draw()
	plt.tight_layout()
	#plt.savefig("cm.png", dpi=300)
	plt.pause(0.001)
	input("Press [Enter] to continue.")
	plt.close()

def saveData():
	"""
	Saves the data of the current sessions
	"""
	global MODEL
	file = open(SESSION_DATA, "wb")
	pickle.dump([MODEL, train_docs, train_classes, test_docs, test_classes, topics], file)
	file.close()

def loadData():
	"""
	Loads the data from the SESSION_DATA
	"""
	global MODEL
	global train_docs
	global test_docs
	global train_classes
	global test_classes
	global topics
	file = open(SESSION_DATA, "rb")
	MODEL, train_docs, train_classes, test_docs, test_classes, topics = pickle.load(file)


def cleanHTML(url):
	"""
	Takes a URL string of a web page as input and return the text from the web page
	along with the title of the page
	"""
	scraper = create_scraper()	
	
	try:
		scrap = scraper.get(url, timeout=5)
		parsed_data = BS(scrap.text, 'html.parser')
		title = "".join(c for c in parsed_data.title.text if c in VALID)
		title = title.strip(" ")
		if len(title) < 1:
			title = "doc_str_" + str(randint(1,10000))
		description = parsed_data.find('meta', attrs={'name': 'description'})
		
		description = ""
		if "content" in str(description):
			description = description.get("content")
			
		h1_content = ""
		h1_list = parsed_data.find_all('h1')
		for i in range (len(h1_list)):
			h1_content = h1_content + h1_list[i].text
			if i != len(h1_list)-1:
				h1_content += ". "

		p_content = ""
		p_list = parsed_data.find_all('p')
		for i in range (len(p_list)):
			p_content = p_content + p_list[i].text
			if i != len(p_list)-1:
				p_content += ". "
		
		h2_content = ""
		h2_list = parsed_data.find_all('h2') 
		for i in range (len(h2_list)):
			h2_content = h2_content + h2_list[i].text
			if i != len(h2_list)-1:
				h2_content += ". "

		h3_content = ""
		h3_list = parsed_data.find_all('h3')
		for i in range (len(h3_list)):
			h3_content = h3_content + h3_list[i].text
			if i !=  len(h3_list)-1:
				h3_content += ". "

		content = str(title) + "\n" + str(description) + "\n" + str(h1_content) + " " + str(h2_content) + " " + str(h3_content) + " " + str(p_content)
		lines = (line.strip() for line in content.splitlines())
		chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
		content = '\n'.join(chunk for chunk in chunks if chunk)
		return content, str(title)
	
	except:
		return "", ""

class HTMLParse(htmlparser):
	"""
	HTMLParse object for the processing of HTML documents
	"""
	def __init__(self):
		self.title = None
		self.rec = not True
		htmlparser.__init__(self)

	def handle_data(self, data):
		if self.rec:
			self.title = data
			
	def handle_starttag(self, tag, attributes):
		if tag == 'title':
			self.rec = not False
		elif tag == 'a':
			attributes = dict(attributes)
			LINKS.append(attributes)

	def handle_endtag(self, tag):
		if tag == 'title':
			self.rec = not True

def getSearchResults(url):
	"""
	Helper function for initDataset
	"""
	r = request.Request(url)

	try:
		response = request.urlopen(r)
	except:
		# Exceptions are ignored
		pass
	else:
		# Scrape all links found from the search using the htmlparser
		html = response.read().decode("utf8")
		parser = HTMLParse()
		parser.feed(html)
		
		# Get links from top related searches
		soup = BS(html, 'html.parser')
		masterURL = "https://ca.search.yahoo.com/search?p={}"
		related = soup.find('ol',class_='cardReg searchTop').text.split(':')[1].split(',')
		
		# For bottom related searches
		#for result in soup.select('.pl-18'):
		#	print(result.text)
		
		for relation in related:
			relation = relation.strip()
			URL = masterURL.format(relation.replace(" ","+"))
			r = request.Request(URL)
		
			try:
				response = request.urlopen(r)
			except:
				pass
			else:
				rlvt = response.read().decode("utf8")
				parser.feed(rlvt)
				
		# Iterate through all the URLS from search results  and create a list of relevant URLS
		links = []
		for link in LINKS:
			if "href" in link.keys():
				if "verizon" not in link['href'] and "#" not in link['href'] and "tumblr" not in link['href'] and "yahoo" not in link['href']:
					links.append(link["href"])
		
		# Purge any duplicate links
		links = list(set(links))
		
		return links

def showDatasetInfo():
	"""
	Displays dataset information
	"""
	if not os.path.exists(DATA_DIR):
		print("[Error]: A dataset has not been initialized!")
		return
	
	subdirs = os.listdir(DATA_DIR)
	topic_count = len(subdirs)
	total_docs = 0
	print("Dataset:")
	
	for _dir in subdirs:
		_subdir = os.path.join(DATA_DIR, _dir)
		doc_count = len(os.listdir(_subdir))
		total_docs += doc_count
		print("\t{:<25}: {:>3} documents".format(_dir, doc_count))
	print()
	print("\tTotal Topics: {}".format(topic_count))
	print("\tTotal Documents: {}".format(total_docs))
	print("\tAverage Topic Size: {:.0f} documents".format(total_docs/topic_count))
	
	_docs, _classes = readDataset(subdirs)
	temp = model(topics=subdirs, stop_words=stopwords.words('english'), min_df=0, lowercase=True)
	temp.train(_docs, _classes)
	print("\tTotal Word Count: {} words".format(temp.collection.shape[1]))

def createDataset(links, folder):
	"""
	Helper function for initDataset
	"""
	for link in links:
		try:
			output, title = cleanHTML(link)
			
			# Ignore any documents without a title
			if len(title) < 1:
				continue
			
			# Ignore any document with less than MIN_CHAR_IN_DOC chars
			if len(output) < MIN_CHAR_IN_DOC:
				continue
			
			if not os.path.exists(DATA_DIR):
				os.makedirs(DATA_DIR)
				
			subDir = os.path.join(DATA_DIR, folder)
			if not os.path.exists(subDir):
				os.makedirs(subDir)
				
			filepath = os.path.join(DATA_DIR, subDir, title+".txt")
			outFile = open(filepath, 'w+', encoding='utf-8')
			outFile.write(output)
			outFile.close()

			print("File added: ", os.path.join(title+".txt"))
		except:
			pass

def initDataSet():
	"""
	Initializes a dataset using a yahoo search query from the list of topics found in topics.txt
	"""
	# Check if dataset already exists, if so prompt user for confirmation of re-initialization
	if os.path.exists(DATA_DIR):
		ans = ""
		while ans not in ["y", "yes", "n", "no"]:
			print("A dataset appears to have been already initialized.")
			ans = input("Reinitialize dataset (y/n)? ")
			ans = ans.lower()
		if ans in ["n", "no"]:
			return
	
	# Iterate through the topics and run Yahoo! search queries on each
	topics = open(TOPICS_FILE_NAME, 'r').read().splitlines()
	masterURL = "https://ca.search.yahoo.com/search?p={}"
	for topic in topics:
		print("Retrieving data for: " + topic)
		URL = masterURL.format(topic.replace(" ","+"))
		links = getSearchResults(URL)
		createDataset(links, topic)
		LINKS.clear()


def fileReader(filename, folder=DATA_DIR):
	"""
	Helper function for the reading of datums from the dataset
	"""
	filepath = os.path.join(folder, filename)
	with open(filepath, 'r', encoding='latin-1') as f:
		return f.read()

def readDataset(topics):
	"""
	Reads the dataset and return the data to be processed
	"""
	docs = []
	classes = []

	for _, topic in enumerate(topics):
		topic_path = os.path.join(DATA_DIR, topic)
		files = os.listdir(topic_path)
		read = partial(fileReader, folder=topic_path)
		content = [read(file) for file in files]
		docs += content
		classes += [topic] * len(content)

	return docs, classes

def getTopicURL():
	"""
	Uses the generated model to predict topics of a user input URL web page
	"""
	if MODEL is None:
		print("[Error]: A model has not been initialized")
		return 
	
	url = input("Enter URL: ")
	
	check = urlparse(url)
	if all([check.scheme, check.netloc]) is False:
		print("[Error]: Invalid URL entered. (URL should start with https:// or http://)")
		return
	
	content, title = cleanHTML(url)
	test = [content]
	predicted_classes = MODEL.predict(test)
	print("Page Title:", title)
	print("Predicted Topic:", predicted_classes[0])

def getTopicList():
	"""
	Uses the generated model to predict topics from URLs found in URL_LIST
	"""
	if MODEL is None:
		print("[Error]: A model has not been initialized")
		return 
	
	url_list = open(URL_LIST, 'r').read().splitlines()
	
	for url in url_list:
		check = urlparse(url)
		if all([check.scheme, check.netloc]) is False:
			continue
		
		content, title = cleanHTML(url)
		test = [content]
		predicted_classes = MODEL.predict(test)
		print("Page Title:", title)
		print("Predicted Topic:", predicted_classes[0])
		print()

def displayModelInfo():
	"""
	Displays information related to the generated model
	"""
	if MODEL is None:
		print("[Error]: A model has not been initialized")
		return 

	predicted_classes = MODEL.predict(test_docs)
	
	# Generate a report showing the main classification metrics
	print("Training Set ({:.0f}% of dataset): {} documents".format((TRAIN_SET_SIZE)*100, len(train_docs)))
	print("Test Set ({:.0f}% of dataset): {} documents".format((1-TRAIN_SET_SIZE)*100, len(test_classes)))
	print()
	print(classification_report(test_classes, predicted_classes))
	accuracy = accuracy_score(test_classes, predicted_classes)
	print("{}: {:.2f} %".format("Global Accuracy", accuracy * 100))
	
	# Get inter-rater reliability for comparison
	kappa_stat = kp(list(map(topics.index, test_classes)), list(map(topics.index, predicted_classes)))
	print("{}: {:.2f} %".format("Kappa Statistic", kappa_stat * 100))
	
	# Get Hamming Loss for comparison
	print("Hamming Loss: {:.2f} %".format(hamming_loss(test_classes, predicted_classes) * 100))
	
	# Get Matthews Correlation Coefficient
	print("MCC: {:.2f}".format(matthews_corrcoef(test_classes, predicted_classes)))
	
	displayConfusionMatrix()


def initModel():
	"""
	Initializes and trains the topic prediction model
	"""
	if not os.path.exists(DATA_DIR):
		print("[Error] A dataset has not been initialized!")
		return

	global MODEL
	global train_docs
	global test_docs
	global train_classes
	global test_classes
	global topics
	
	if MODEL is not None:
		ans = ""
		while ans.lower() not in ["y", "yes", "n", "no"]:
			print("A model appears to have been already initialized.")
			ans = input("Reinitialize model (y/n)? ")
			ans = ans.lower()
		if ans.lower() in ["n", "no"]:
			return
	
	topics = os.listdir(DATA_DIR)
	
	# Split data to test and train
	documents, classes = readDataset(topics)
	train_docs, test_docs, train_classes, test_classes = dataset_split(documents, classes, train_size=TRAIN_SET_SIZE)

	# Initialize the model
	# When building the vocabulary ignore terms that have a document frequency strictly lower than 1
	# Convert all characters to lowercase before tokenizing.
	# Filter stopwords from the resulting tokens.
	MODEL = model(topics=topics, stop_words=stopwords.words('english'), min_df=2, lowercase=True)
	
	# Train the model using the split data
	MODEL.train(train_docs, train_classes)
	
	saveData()
	# Display results
	print("Training Set ({:.0f}% of dataset):".format(TRAIN_SET_SIZE*100))
	print("{}: {} documents".format("\tTotal", len(train_docs)))
	print("{}: {} words".format("\tWord Count", MODEL.collection.shape[1]))
	print("[Info]: Model successfully initialized and trained.")

def displayMenu():
	"""
	Displays the menu
	"""
	print("Select an option:")
	print("\t1. Initialize Dataset")
	print("\t2. View Dataset Information")
	print("\t3. Initialize and Train Model")
	print("\t4. View Model Assessment Report")
	print("\t5. Predict Topic from a URL")
	print("\t6. Predict Topics from URLs in {}".format(URL_LIST))
	print("\t7. Exit")
	
def main():
	"""
	Main function
	"""
	print("#"*45)
	print("{:>30}".format("WEB PAGE CLASSIFIER"))
	print("#"*45)
	
	if os.path.isfile(SESSION_DATA):
		loadData()
		print("[INFO]: Data loaded from [{}]".format(SESSION_DATA))
	
	choice = 0
	displayMenu()
	try:
		while (choice != 7):
			action_input = input("Select: ")
			if action_input.isdigit() is False:
				print("Invalid action")
				continue;
			choice = int(action_input)
			if choice < 1 or choice > 7:
				print("Invalid action")
				choice = 0
				continue
			
			# Initialize Dataset
			if choice == 1:
				print(SEPARATOR)
				initDataSet()
				print(SEPARATOR)
				displayMenu()
				continue
	
			# Display Dataset
			if choice == 2:
				print(SEPARATOR)
				showDatasetInfo()
				print(SEPARATOR)
				displayMenu()
				continue
			
			# Initialize and Train Model
			if choice == 3:
				print(SEPARATOR)
				initModel()
				print(SEPARATOR)
				displayMenu()
				continue
			
			# Display Model details
			if choice == 4:
				print(SEPARATOR)
				displayModelInfo()
				print(SEPARATOR)
				displayMenu()
				continue
			
			# Predict topic from URL
			if choice == 5:
				print(SEPARATOR)
				getTopicURL()
				print(SEPARATOR)
				displayMenu()
				continue
			
			# Predict topics from URL_LIST
			if choice == 6:
				print(SEPARATOR)
				getTopicList()
				print(SEPARATOR)
				displayMenu()
				continue
	except:
		print()
		
	print("[Program Terminated]")
	
if __name__ == "__main__":
	main()
