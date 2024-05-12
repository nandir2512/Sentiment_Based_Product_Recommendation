# Project Overview: Sentiment Based Product Recommendation System

### Folder Structure:
* Data: Contains the source data for the project.
* Templates: Houses the HTML file (index.html) responsible for the user interface.
* app.py: Flask application file responsible for running the server and handling user requests.
* model.py: Contains the code for sentiment analysis and user-based recommendation system.
* Sentiment Based Product Recommendation System.ipynb: Jupyter Notebook containing the project code, which can be executed step by step.

### Execution Steps:
1. Sentiment Based Product Recommendation System.ipynb:
    * Open the Jupyter Notebook Sentiment Based Product Recommendation System.ipynb.
    * Execute each cell sequentially by either clicking on the cell and pressing (Shift + Enter) or using the "Run" button in the toolbar.
    * This notebook contains code for data preprocessing, model training (including logistic regression for sentiment analysis and user-based recommendation system), and evaluation.

2. app.py:
    * Once you've executed the notebook and saved the trained models, run app.py.
    * This will launch a local server where users can interact with the recommendation system through a web interface.
    * Users can input their name, and the system will provide them with personalized product recommendations based on sentiment analysis and user behavior.

## User Guide:
* To use the recommendation system, navigate to the project folder in your terminal or command prompt.
* Make sure you have all dependencies installed (pip install -r requirements.txt).
* Execute app.py by running python app.py in your terminal.
Once the server is running, open your web browser and go to http://localhost:5000/.
* You will be greeted with a simple user interface prompting you to enter your name.
* After entering your name and submitting the form, the system will process the information and provide you with personalized product recommendations.
* Explore the recommendations and interact with the system to discover products tailored to your preferences and sentiment.

# Problem Statement
The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

 
Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

 
With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

 
As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 


In order to do this, you planned to build a sentiment-based product recommendation system, which includes the following tasks.

*  Data sourcing and sentiment analysis
*  Building a recommendation system
*  Improving the recommendations using the sentiment analysis model
*  Deploying the end-to-end project with a user interface
 

##### Data sourcing and sentiment analysis
In this task, you have to analyse product reviews after some text preprocessing steps and build an ML model to get the sentiments corresponding to the users' reviews and ratings for multiple products. 

This dataset consists of 30,000 reviews for more than 200 different products. The reviews and ratings are given by more than 20,000 users. Please refer to the following attribute description file to get the details about the columns of the Review Dataset.


##### The steps to be performed for the first task are given below.
* Exploratory data analysis
* Data cleaning
* Text preprocessing
* Feature extraction: In order to extract features from the text data, you may choose from any of the methods, including bag-of-words, TF-IDF vectorization or word embedding.
* Training a text classification model: You need to build at least three ML models. You then need to analyse the performance of each of these models and choose the best model. At least three out of the following four models need to be built (Do not forget, if required, handle the class imbalance and perform hyperparameter tuning.). 
    1. Logistic regression
    2. Random forest
    3. XGBoost
    4. Naive Bayes

Out of these four models, you need to select one classification model based on its performance.

#### Building a recommendation system
As you learnt earlier, you can use the following types of recommendation systems.
    1. User-based recommendation system
    2. Item-based recommendation system
 

Your task is to analyse the recommendation systems and select the one that is best suited in this case. 

 

Once you get the best-suited recommendation system, the next task is to recommend 20 products that a user is most likely to purchase based on the ratings. You can use the 'reviews_username' (one of the columns in the dataset) to identify your user. 

 

#### Improving the recommendations using the sentiment analysis model
Now, the next task is to link this recommendation system with the sentiment analysis model that was built earlier (recall that we asked you to select one ML model out of the four options). Once you recommend 20 products to a particular user using the recommendation engine, you need to filter out the 5 best products based on the sentiments of the 20 recommended product reviews. 

 

In this way, you will get an ML model (for sentiments) and the best-suited recommendation system. Next, you need to deploy the entire project publically.

 

#### Deployment of this end to end project with a user interface
Once you get the ML model and the best-suited recommendation system, you will deploy the end-to-end project. You need to use the Flask framework, which is majorly used to create web applications to deploy machine learning models.

 

Next, you need to include the following features in the user interface.

    1. Take any of the existing usernames as input.
    2. Create a submit button to submit the username.
    3. Once you press the submit button, it should recommend 5 products based on the entered username.

Note: An important point that you need to consider here is that the number of users and the number of products are fixed in this case study, and you are doing the sentiment analysis and building the recommendation system only for those users who have already submitted the reviews or ratings corresponding to some of the products in the dataset. 


Assumption: No new users or products will be introduced or considered when building or predicting from the models built.

 

#### What needs to be submitted for the evaluation of the project?

1. An end-to-end Jupyter Notebook, which consists of the entire code (data cleaning steps, text preprocessing, feature extraction, ML models used to build sentiment analysis models, two recommendation systems and their evaluations, etc.) of the problem statement defined 
2. The following deployment files 
    * One 'model.py' file, which should contain only one ML model and only one recommendation system that you have obtained from the previous steps along with the entire code to deploy the end-to-end project using Flask and Heroku
    * 'index.html' file, which includes the HTML code of the user interface
'app.py' file, which is the Flask file to connect the backend ML model with the frontend HTML code
    * Supported pickle files, which have been generated while pickling the models


# Solution: 
### Steps I did: 
* Data Cleaning and Pre-Processing
* Text Preprocessing 
* Feature Extraction using Tfidf Vectorizer
* Model Builded for below Algo.
    * Logistic Model 
    * Decision Tree
    * Random Forest
    * XGBoost
    * Naive Bayes
* Building a Recommendation System
    * User-based recommendation system
    * Item-based recommendation system
* Recommendation of Top 20 Products to a Specified User
* Fine-Tuning the Recommendation System and Recommendation of Top 5 Products

## Comment:
After thorough analysis of various models for sentiment analysis, it was concluded that Logistic Regression exhibited superior performance compared to other models. Its ability to effectively discern sentiment from textual data was evident through its robust metrics and consistent performance across sensitivity, specificity, AUC, and F1 score.

Additionally, in the realm of recommendation systems, the user-based recommendation approach emerged as the most fitting choice. This methodology, leveraging user behavior and preferences to generate personalized recommendations, offers a tailored user experience, thereby enhancing engagement and satisfaction.

By adopting Logistic Regression for sentiment analysis and employing a user-based recommendation system for product recommendations, we are poised to deliver a comprehensive and optimized solution that caters to both sentiment analysis and personalized recommendations, thereby maximizing user engagement and satisfaction.