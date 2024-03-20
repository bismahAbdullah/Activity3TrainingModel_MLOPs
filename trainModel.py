
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset
data = pd.read_csv("onlinefoods.csv")

# Data preprocessing
data.dropna(inplace=True)  # Drop rows with missing values
X = data.drop(columns=["Output", "Feedback"])  # Features
y = data["Output"]  # Target variable

# Encode categorical variables
encoder = LabelEncoder()
for column in ["Gender", "Marital Status", "Occupation"]:
    X[column] = encoder.fit_transform(X[column])

# Encode "Monthly Income" categories
income_map = {"No Income": 0, "Below Rs.10000": 1}
X["Monthly Income"] = X["Monthly Income"].map(income_map)

# Encode "Educational Qualifications" categories
edu_map = {"Graduate": 0, "Post Graduate": 1}
X["Educational Qualifications"] = X["Educational Qualifications"].map(edu_map)

# Encode "Pin code" categories
pincode_map = {"Yes": 1, "No": 0}
X["Pin code"] = X["Pin code"].map(pincode_map)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Preprocess incoming data
    input_data = pd.DataFrame(data, index=[0])
    for column in ["Gender", "Marital Status", "Occupation"]:
        input_data[column] = encoder.transform(input_data[column])

    # Encode "Monthly Income" categories
    input_data["Monthly Income"] = input_data["Monthly Income"].map(income_map)
    
    # Encode "Pin code" categories
    input_data["Pin code"] = input_data["Pin code"].map(pincode_map)
    
    # Make prediction
    prediction = clf.predict(input_data)[0]
    feedback = "Positive" if prediction == "Yes" else "Negative"
    
    return jsonify({"prediction": prediction, "feedback": feedback})

if __name__ == '__main__':
    app.run(debug=True)
