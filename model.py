import pickle
import numpy as np

class RecommendationEngine:
    def __init__(self):
        """
        Initialize the RecommendationEngine class by loading the necessary models and data.
        """
        self.recommend_model = pickle.load(open("models/best_recommendation_model.pkl", 'rb'))
        self.tfidf_vectorizer = pickle.load(open("models/tfidf.pkl", 'rb'))
        self.sentiment_model = pickle.load(open("models/lr_base_model.pkl", 'rb'))
        self.df_text = pickle.load(open("models/df_text.pkl", 'rb'))

    def top5recommendation(self, username):
        """
        Get the top 5 recommended products based on the given username.

        Args:
            username (str): The username for which to get recommendations.

        Returns:
            pandas.DataFrame: A DataFrame containing the top 5 recommended products and their sentiment analysis scores.
        """
        # Check if the username is valid
        if username not in self.recommend_model.index:
            print(f"The User {username} is not a valid user name.")
            return None

        # Get the top 20 recommended products for the user
        top20_recommendations = list(self.recommend_model.loc[username].sort_values(ascending=False)[0:20].index)
        top20_products = self.df_text[self.df_text.id.isin(top20_recommendations)]

        # Perform sentiment analysis on the top 20 products
        X = self.tfidf_vectorizer.transform(top20_products["lema_review"].values.astype(str))
        top20_products['predicted_sentiment'] = self.sentiment_model.predict(X)

        # Calculate the sentiment score for each product
        pred_df = top20_products.groupby(by='name').sum()
        pred_df.rename(columns={'predicted_sentiment': 'pos_sent_count'}, inplace=True)
        pred_df['total_sent_count'] = top20_products.groupby(by='name')['predicted_sentiment'].count()
        pred_df['post_sent_percentage'] = np.round(pred_df['pos_sent_count'] / pred_df['total_sent_count'] * 100, 2)

        # Get the top 5 recommended products based on the sentiment score
        result = pred_df.sort_values(by='post_sent_percentage', ascending=False)[:5].index
        return result