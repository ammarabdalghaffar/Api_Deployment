from flask import Flask, jsonify, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

app = Flask(__name__)

# Load the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
whole_predection = {"type": "t-shirt", "master_category": "Apparel", "gender": "men"}
types_map = {
    0: "Shirts", 1: "Jeans", 2: "Track Pants", 3: "Socks", 4: "Casual Shoes", 5: "Flip Flops", 6: "Sandals",
    7: "Sweatshirts", 8: "Formal Shoes", 9: "Flats", 10: "Sports Shoes", 11: "Shorts", 12: "Heels", 13: "Scarves",
    14: "Rain Jacket", 15: "Dresses", 16: "Night suits", 17: "Skirts", 18: "Blazers", 19: "Backpacks", 20: "Caps",
    21: "Dresses", 22: "Jackets", 23: "Lounge Pants", 24: "Sports Sandals", 25: "Sweaters", 26: "Tracksuits",
    27: "Swimwear", 28: "Nightdress", 29: "Leggings", 30: "Robe", 31: "Lounge Tshirts", 32: "Lounge Shorts",
    33: "Jeggings", 34: "Clothing Set", 35: "Hat", 36: "Trousers", 37: "Suits",
}

@app.route('/get_data', methods=['POST'])
def get_data():
    with open('C:/Users/Sroor For Laptop/OneDrive - Elshorouk Academy/Microsoft Teams Data/Desktop/knn_model.pickle', 'rb') as f:
        gender_loaded_model = pickle.load(f)
    gender_data = request.data.decode('utf-8')
    print("######################################################")
    print(gender_data)
    sample_input = [json.loads(gender_data).get("data")]

    with open('C:/Users/Sroor For Laptop/OneDrive - Elshorouk Academy/Microsoft Teams Data/Desktop/Project/tfidf_vectorizer (1).pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    X_tfidf = tfidf_vectorizer.transform(sample_input)
    print(X_tfidf)
    gender_prediction = gender_loaded_model.predict(X_tfidf)
    print(gender_prediction)
    if gender_prediction == 0:
        whole_predection["gender"] = "Boys"
    elif gender_prediction == 1:
        whole_predection["gender"] = "Girls"
    elif gender_prediction == 2:
        whole_predection["gender"] = "Men"
    elif gender_prediction == 3:
        whole_predection["gender"] = "Unisex"
    elif gender_prediction == 4:
        whole_predection["gender"] = "Women"

    print(whole_predection)
    print("######################################################")

    with open('C:/Users/Sroor For Laptop/OneDrive - Elshorouk Academy/Microsoft Teams Data/Desktop/Project/knn_category_model (2).pkl', 'rb') as f:
        master_loaded_model = pickle.load(f)
    master_data = request.data.decode('utf-8')
    print(master_data)
    sample_input = [json.loads(master_data).get("data")]

    with open('C:/Users/Sroor For Laptop/OneDrive - Elshorouk Academy/Microsoft Teams Data/Desktop/Project/tfidf_vectorizer (1).pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    X_tfidf = tfidf_vectorizer.transform(sample_input)
    print(X_tfidf)
    master_prediction = master_loaded_model.predict(X_tfidf)
    print(master_prediction)
    if master_prediction == 0:
        whole_predection["master_category"] = "Accessories"
    elif master_prediction == 1:
        whole_predection["master_category"] = "Apparel"
    elif master_prediction == 2:
        whole_predection["master_category"] = "Footwear"
    elif master_prediction == 3:
        whole_predection["master_category"] = "Free Items"

    with open('C:/Users/Sroor For Laptop/OneDrive - Elshorouk Academy/Microsoft Teams Data/Desktop/Project/knn_model_type (1).pickle', 'rb') as f:
        type_loaded_model = pickle.load(f)
    type_data = request.data.decode('utf-8')
    print(type_data)
    sample_input = [json.loads(type_data).get("data")]

    with open('C:/Users/Sroor For Laptop/OneDrive - Elshorouk Academy/Microsoft Teams Data/Desktop/Project/tfidf_vectorizer (1).pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    X_tfidf = tfidf_vectorizer.transform(sample_input)
    print(X_tfidf)
    type_prediction = type_loaded_model.predict(X_tfidf)
    print(type_prediction)
    whole_predection["type"] = types_map[type_prediction[0]]

    return jsonify(whole_predection)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
