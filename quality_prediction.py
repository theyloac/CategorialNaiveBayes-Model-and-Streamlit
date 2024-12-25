import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Set up the Streamlit web app configurations:
# - page_title: The title on the browser tab.
# - layout: Ensures the width of the content is stretched.
st.set_page_config(
    page_title="Vehicle Classification",
    layout="wide"
)

@st.cache_data
def load_data_and_model():
    """
    This function loads the car dataset, preprocesses it,
    trains a Categorical Naive Bayes model, and returns
    the encoder, model, accuracy, and the original dataframe.

    The @st.cache_data decorator caches the results, so
    when called again, it won't re-run unless the function
    code or data changes.
    """
    # 1. Load the dataset from a CSV file
    #    The file 'car.csv' must be in the same directory or specify full path.
    cars = pd.read_csv("car.csv", sep=",")

    # 2. Create an OrdinalEncoder instance to handle categorical encoding
    encoder = OrdinalEncoder()

    # 3. Convert all feature columns (excluding 'class') to 'category' datatype
    for col in cars.columns.drop('class'):
        cars[col] = cars[col].astype('category')

    # 4. Encode the feature columns (drop 'class' because that's the target)
    X_encoded = encoder.fit_transform(cars.drop('class', axis=1))

    # 5. Convert target column 'class' to numerical codes
    #    .cat.codes automatically converts each category to an integer code.
    y = cars['class'].astype('category').cat.codes

    # 6. Split data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,    # Encoded feature matrix
        y,            # Target array
        test_size=0.3,
        random_state=42
    )

    # 7. Initialize and train the Categorical Naive Bayes model
    model = CategoricalNB()
    model.fit(X_train, y_train)

    # 8. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 9. Compute the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # 10. Return everything needed for prediction and display
    return encoder, model, accuracy, cars

# Call the function to load data and model
encoder, model, accuracy, cars = load_data_and_model()

# Set a title for the Streamlit app
st.title("Vehicle Quality Prediction")

# Display the model's accuracy
st.write(f"Model accuracy: {accuracy:.2f}")

# Create a list of input widgets (selectboxes) to capture user input
# Each selectbox allows the user to pick from the categorical values in the data
input_features = [
    st.selectbox("Price:", cars['buying'].unique()),
    st.selectbox("Maintenance:", cars['maint'].unique()),
    st.selectbox("Doors:", cars['doors'].unique()),
    st.selectbox("Passenger Capacity:", cars['persons'].unique()),
    st.selectbox("Trunk Space:", cars['lug_boot'].unique()),
    st.selectbox("Safety:", cars['safety'].unique()),
]

# When the user clicks the "Process" button, we:
# 1. Convert the user inputs to a DataFrame.
# 2. Encode the user input with the same encoder used during training.
# 3. Use the trained model to predict the class.
# 4. Convert the prediction back to its original label.
# 5. Display the predicted result to the user.
if st.button("Process"):
    # Turn the list of features into a single-row DataFrame, matching training columns
    input_df = pd.DataFrame([input_features], columns=cars.columns.drop('class'))
    
    # Encode the user input
    input_encoded = encoder.transform(input_df)
    
    # Get the numeric prediction from the model
    prediction_encoded = model.predict(input_encoded)
    
    # Decode the numeric prediction back to its categorical label
    prediction = cars['class'].astype('category').cat.categories[prediction_encoded][0]
    
    # Display the result on the interface
    st.header(f"Prediction Result:  {prediction}")
