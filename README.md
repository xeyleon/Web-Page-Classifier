# Web-Page-Classifier
A web page classifier implemented using Python 3. The program provides a terminal, menu-based interaction to the user.

A supervised learning approach is adopted, using a dataset (corpus) consisting of numerous example pages from each topic, for classifier modeling. Although its accuracy is not optimal, due to its speed and versatility, the Naive Bayes classifier model is employed, along with Bag-of-Words (BOW) for feature extraction. To this end, the Natural Language Processing Kit (NLT) and Scikit-Learn Python modules are heavily utilized.

## Installation

The program is written in Python 3, so Python 3 must be installed for its usage.

To ensure problem-free execution of the program, a requirements.txt is provided, which lists all the necessary packages utilized by the program.

Assuming one has a terminal open inside the directory of the program, all prerequisite packages can be installed using running following command: `pip install -r requirements.txt`

The program is executed using the following command: `python web_classifier.py` (The prescribed commands may vary depending on the user’s system environment.)

##  Program Specifications
* The size of the training set can be set in the settings of the code (a 70% training set is the default).
* The listed topics within the `topics.txt` are the topics that the model is trained with and will recognize.
* The minimum number of characters permitted for a dataset document can be set in the settings of the code (the default is 80).
* The webcrawler is built-in to the program, and is utilized for dataset initialization.
* During dataset initialization, the web crawler retrieves the first page search results from Yahoo! search engine, as clean text documents, for each topic found in `topics.txt`.
* The session data is retained, such that once the user has initialized and trained a model, the trained model and relevant data used for modeling is stored, and if present, automatically restored on the next execution of the program.

## Program Features
 Upon execution of the program, the user is presented the following menu of options to select (omitting the descriptions):

 ### 1. Initialize a Dataset
  * Using the list of topics found in the topics.txt, the program creates a dataset by retrieving the first page search results from the Yahoo! search engine for each topic.
  * This dataset acts as the corpus from which the model will be trained.
  * The dataset must be initialized before any of the other actions offered by the program can be performed.

### 2. View Dataset Information
  * The user is presented with details of the dataset, such as the number of topics, and the number of documents existing for each topic.
  * Cannot be run without a dataset existing or having been initialized.

### 3. Initialize and Train Model
  * Utilizing the dataset, the classification model is initialized and trained using a subset of the dataset as a training set. The size of the training set is defined at the header settings section of the code.
  * Cannot be run without a dataset existing or having been initialized.

### 4. View Model Assessment Report
  * The user is presented with a classification report of the model and confusion matrix using the test set, along with other relevant information.
  * Cannot be run without a model having been initialized.

### 5. Predict Topic from a URL
  * The topic of the page found at a user input URL is predicted by the trained model.
  * The input URL should start with https:// or https://
  * Cannot be run without a model having been initialized

### 6. Predict topics from URL in (default: `url_list.txt`)
  * The topic of the page found at each URL found in the text is predicted by the trained model.
  * Invalid URLs are ignored.
  * Cannot be run without a model having been initialized
