# ML-Mini-Project-Sai-Rahul-CSE-B-3122215001090-Sentiment-analysis-of-E-commerce-website
Sentiment analysis of E-commerce website 
Introduction:
A Sentiment Analysis is the process of analysing digital text to determine the emotional tone of the message is positive, negative or neutral. Sentiment analysis tools can scan this text to automatically determine the reviewer’s attitude towards a topic. Companies use the insights from sentiment analysis to improve customer service and increase brand reputation.
The provided code in this project, performs sentiment analysis on a dataset containing reviews from an e-commerce platform, (Flipkart). The aim is to analyze customer sentiments based on their reviews and classify them as positive or negative.

Objective:
The primary objective of this analysis is to develop a sentiment analysis model that accurately classifies reviews as positive or negative. By doing so, we can gain insights into customer sentiments and feedback, which can be valuable for businesses to understand customer satisfaction and make data-driven decisions.

Analysis:
Data Loading and Exploration:
The code begins by loading the dataset flipkart_data.csv using pandas.
It explores the unique ratings and visualizes the distribution of ratings using a countplot.

Data Preprocessing:

The ratings are converted into binary labels (0 for negative and 1 for positive) based on a threshold (5).
Text preprocessing is performed using the pre process text function, which removes punctuations, converts text to lowercase, and removes stopwords using NLTK.

Word Cloud Visualization:
A word cloud is generated for positive reviews to visualize frequently occurring words.
This provides a graphical representation of the most common words in positive reviews.

Feature Extraction (TF-IDF):
TF-IDF (Term Frequency-Inverse Document Frequency) features are extracted from the review text using TfidfVectorizer.
This step converts text data into numerical features that can be used for training machine learning models.

Model Building:
The dataset is split into training and testing sets using train_test_split.
A decision tree classifier is trained on the TF-IDF features using DecisionTreeClassifier.

Model Evaluation:
The trained model is tested on the training data.
The confusion matrix is computed to evaluate the performance of the model.
The confusion matrix is visualized using ConfusionMatrixDisplay.

Algorithm:
•	import Libraries:
•	Import necessary libraries for the project.
•	warnings library is used to ignore any warnings.
•	pandas is used for data manipulation and analysis.
•	re is used for regular expression operations.
•	seaborn is used for data visualization.
•	sklearn is used for machine learning tasks.
•	matplotlib is used for plotting graphs.
•	WordCloud is used for generating word clouds.
•	nltk is used for natural language processing tasks.
•	tqdm is used for showing progress bars.


Reading Data:
Reading the dataset flipkart_data.csv using pd.read_csv() into a DataFrame called data.
Displaying the first few rows of the dataset using data.head().

Counting Unique Ratings:
Using pd.unique(data['rating']) to get the unique values of the 'rating' column.
Visualizing the count of each rating using sns.countplot().

Creating Label (Binary Classification):
Creating a new column 'label' based on the 'rating' column.
If the 'rating' is greater than or equal to 5, assigning 1 (positive), else 0 (negative).

Preprocessing Text:
Defining a function preprocess_text to preprocess the text data.
Removing punctuations using re.sub(r'[^\w\s]', '', sentence).
Converting text to lowercase and removing stopwords using NLTK's stopwords.

Visualizing Positive Reviews Word Cloud:
Creating a word cloud for positive reviews ('label' = 1) using the WordCloud library.
Displaying the word cloud using plt.imshow() and plt.show().

Creating TF-IDF Features:
Using TfidfVectorizer to convert text data into TF-IDF features.
Limiting the number of features to 2500 using max_features=2500.


Splitting Data into Train and Test Sets:
Splitting the data into training and testing sets using train_test_split() from sklearn.
Using 33% of the data for testing and 67% for training.
Using stratify=data['label'] to maintain the ratio of positive and negative labels in train and test sets.

Training Decision Tree Classifier:
Creating a decision tree classifier using DecisionTreeClassifier() from sklearn.
Fitting the model on the training data using fit().

Testing the Model:
Predicting labels for the training data using predict() and storing it in pred.
Printing the actual labels (y_train) and predicted labels (pred).

Confusion Matrix:
Computing the confusion matrix using confusion_matrix() from sklearn.
Displaying the confusion matrix using ConfusionMatrixDisplay and plot() from sklearn.

Conclusion:
The sentiment analysis model developed in this analysis demonstrates the potential to classify reviews as positive or negative based on customer feedback. By leveraging natural language processing techniques and machine learning algorithms, businesses including ecommerce websites, can gain valuable insights into customer sentiments, identify areas for improvement, and enhance overall customer satisfaction

