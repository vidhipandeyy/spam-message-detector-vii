import pickle

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Take input message
message = input("Enter a message: ")

# Convert message to numbers
message_vectorized = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vectorized)

# Show result
if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")