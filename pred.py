import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to extract Mel spectrogram features from an audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

# Function to load and preprocess audio files from a folder
def load_audio_files_from_folder(folder_path):
    file_paths = []
    labels = []
    print(f"Loading files from folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)
                print(f"Found file: {file_path}")
                # Assuming the file name format is <label>_<anything>.wav
                # where <label> is either '0' or '1'
                try:
                    label = int(file_name.split('_')[0])
                    labels.append(label)
                except ValueError:
                    print(f"Skipping file with invalid label format: {file_name}")
    features = []
    for file_path in file_paths:
        mel_spectrogram = extract_features(file_path)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
        features.append(mel_spectrogram)
    print(f"Loaded {len(features)} files with {len(labels)} labels.")
    return np.array(features), np.array(labels)

# Define the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to load the model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict using the extracted features and the model
def predict_suicidal_tendencies(audio_path, model):
    mel_spectrogram = extract_features(audio_path)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Reshape for prediction
    prediction = model.predict(mel_spectrogram)
    return prediction[0][0]

# Function to predict for all files in a folder
def predict_folder(folder_path, model):
    predictions = {}
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.wav'):
                audio_path = os.path.join(root, file_name)
                prediction = predict_suicidal_tendencies(audio_path, model)
                predictions[file_name] = prediction
    return predictions

# Function to plot the results
def plot_predictions(predictions):
    plt.figure(figsize=(10, 6))
    file_names = list(predictions.keys())
    probs = list(predictions.values())
    plt.bar(file_names, probs)
    plt.ylabel('Probability')
    plt.title('Suicidal Prediction for Each Audio File')
    plt.xticks(rotation=90)
    plt.show()

# Load your dataset from the folder and train the model
def train_and_save_model(folder_path, model_save_path):
    X, y = load_audio_files_from_folder(folder_path)
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("No data loaded. Please check the folder path and file format.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = create_cnn_model(input_shape)
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')

    # Save the trained model in .h5 format
    model.save(model_save_path)
    print(f'Model saved at {model_save_path}')

# Example usage
training_folder_path = 'C:/Users/pghar/Downloads/Audio_Speech_Actors_01-24'  # Replace with the path to your training audio folder
model_save_path = 'C:/Users/pghar/OneDrive/Desktop/spltopics/harshitha/suicidal_prediction_model.h5'  # Replace with the path to save your trained model

# Train the model and save it
train_and_save_model(training_folder_path, model_save_path)

# Predict using the saved model
predict_folder_path = 'C:/Users/pghar/Downloads/Audio_Speech_Actors_01-24/Actor_24/'  # Replace with the path to your audio folder for prediction
model = load_model(model_save_path)

predictions = predict_folder(predict_folder_path, model)
for file_name, prediction in predictions.items():
    print(f'{file_name}: {prediction}')
plot_predictions(predictions)

