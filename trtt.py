from flask import Flask, render_template, request
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv('covid_symptoms_data.csv')
data = data.drop(['Country'], axis=1)  # Remove the Country column

# Generate frequent itemsets
frequent_itemsets = apriori(data, min_support=0.4, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

@app.route('/', methods=['GET', 'POST'])
def predict():
    radar_data = None  # Initialize radar_data
    show_graph = False  # Control whether to show the radar chart

    if request.method == 'POST':
        # Check if the "Show Graph" button was pressed
        if 'show_graph' in request.form:
            show_graph = True
            
            # Prepare radar data when the graph is requested
            radar_data = {
                'Fever': int(data['Fever'].sum()),  # Convert to int
                'Tiredness': int(data['Tiredness'].sum()),  # Convert to int
                'Dry-Cough': int(data['Dry-Cough'].sum()),  # Convert to int
                'Difficulty-in-Breathing': int(data['Difficulty-in-Breathing'].sum()),  # Convert to int
                'Sore-Throat': int(data['Sore-Throat'].sum()),  # Convert to int
                'Pains': int(data['Pains'].sum()),  # Convert to int
                'Nasal-Congestion': int(data['Nasal-Congestion'].sum()),  # Convert to int
                'Runny-Nose': int(data['Runny-Nose'].sum()),  # Convert to int
                'Diarrhea': int(data['Diarrhea'].sum())  # Convert to int
            }

        # Handle the "Predict" button press
        elif 'predict' in request.form:
            # Get user input
            symptoms = request.form.getlist('symptoms')
            age = request.form.get('age')
            gender = request.form.get('gender')
            contact = request.form.get('contact')

            # Create a set of user input features
            user_features = set(symptoms + [age, gender, contact])

            # Calculate the probability of having COVID-19
            matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(user_features))]
            
            if not matching_rules.empty:
                # Calculate a weighted average of confidences
                total_weight = 0
                weighted_sum = 0
                for _, rule in matching_rules.iterrows():
                    weight = len(rule['antecedents'])
                    total_weight += weight
                    weighted_sum += rule['confidence'] * weight
                
                probability = weighted_sum / total_weight if total_weight > 0 else 0
                probability_percentage = probability * 100
                
                # Determine severity based on probability
                if probability_percentage <= 70:
                    result = "Patient condition is Mild."
                elif 71 <= probability_percentage <= 80:
                    result = "Patient condition is Moderate."
                else:
                    result = "Condition Severe: Immediate Care Needed."
            else:
                result = "Unable to determine the probability based on the given information."
            
            return render_template('index.html', result=result, radar_data=None, show_graph=False)

    return render_template('index.html', radar_data=radar_data, show_graph=show_graph)

if __name__ == '__main__':
    app.run(debug=True)
