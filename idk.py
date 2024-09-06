
# Function to train the CNN model
def train_model(train_data, train_labels, epochs=10):
    input_shape = train_data.shape[1:]
    model = create_cnn_model(input_shape)
    model.fit(train_data, train_labels, epochs=epochs)
    return model

# Function to predict suicidal tendencies from an audio file
def predict_suicidal_tendencies(audio_path, model):
    mel_spectrogram = extract_features(audio_path)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Reshape for prediction
    prediction = model.predict(mel_spectrogram)
    return prediction[0][0]

# Plot the prediction
def plot_prediction(prediction, file_name):
    plt.figure(figsize=(6, 4))
    plt.bar(['Non-Suicidal', 'Suicidal'], [1-prediction, prediction])
    plt.ylabel('Probability')
    plt.title('Suicidal Tendency Prediction for {}'.format(file_name))
    plt.show()

# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Load the dataset and train the model
# Assuming you have a dataset with features (X) and labels (y)
# Replace X_train, y_train, X_test, and y_test with your actual training and test data
X_train = np.random.random((1000, 128, 128, 1))  # Example dummy data
y_train = np.random.randint(0, 2, 1000)  # Example dummy labels
X_test = np.random.random((200, 128, 128, 1))  # Example dummy test data
y_test = np.random.randint(0, 2, 200)  # Example dummy test labels

model = train_model(X_train, y_train, epochs=10)

# Evaluate the model on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, class_names=['Non-Suicidal', 'Suicidal'])

# Example usage with a single audio file for prediction
audio_file_path = input("Enter the path to the audio file in WAV format: ")  # Path to your audio file
if not os.path.exists(audio_file_path):
    print("Audio file not found. Please provide a valid file path.")
    exit()

# Convert audio file to WAV format if necessary
audio_file_path = convert_to_wav(audio_file_path)

# Predict suicidal tendency
prediction = predict_suicidal_tendencies(audio_file_path, model)
print(f'Predicted probability of suicidal tendency: {prediction}')

# Plot the prediction
plot_prediction(prediction, os.path.basename(audio_file_path))
