from flask import Flask, render_template, request
from model import RecommendationEngine

app = Flask(__name__)
recom_model = RecommendationEngine()

@app.route('/')
def home():
    """
    Render the index.html template for the home page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the prediction request and render the result on the index.html template.
    """
    # Get the username as input
    user_name_input = request.form['username'].lower()

    # Get the top 5 recommended products
    sent_reco_output = recom_model.top5recommendation(user_name_input)

    if sent_reco_output is not None:
        # Render the result on the index.html template
        return render_template("index.html", output=sent_reco_output)
    else:
        # Handle the case when the username is invalid
        return render_template("index.html", message_display="Opps! Wrong User Name. Please try with valid user!")

if __name__ == '__main__':
    app.run(debug=True)