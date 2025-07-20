import pandas as pd
import os
import cv2
import mediapipe as mp
import coll  # Make sure to import the coll module
from sklearn.ensemble import RandomForestClassifier

# ... (your existing imports)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ... (your existing code)

# Define the get_prediction function
def get_prediction(image, clf):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        ImageData = coll.ImageToDistanceData(image, hands)
        DistanceData = ImageData['Distance-Data']
        prediction = clf.predict([DistanceData])
        return prediction[0]

# Function to collect hand gesture data and save to CSV
def collect_and_save_data(units_in_data, clf):
    cap = cv2.VideoCapture(0)

    data = {'unit-' + str(i): [] for i in range(units_in_data)}
    data['letter'] = []

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            try:
                SpelledWord = get_prediction(image, clf)

                # Append features to the data dictionary
                ImageData = coll.ImageToDistanceData(image, hands)
                DistanceData = ImageData['Distance-Data']
                for i in range(units_in_data):
                    data['unit-' + str(i)].append(DistanceData[i])

                # Append the label to the data dictionary
                data['letter'].append(SpelledWord)

            except Exception as e:
                print(f"Error processing frame: {e}")

            cv2.imshow('frame', image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press escape to break
                break

    # Convert the data dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_filename = 'hand_gesture_data.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Dataset saved to {csv_filename}")

    cap.release()
    cv2.destroyAllWindows()

# Specify the number of units in data
units_in_data = 28

# # Create and train the Random Forest classifier
# clf = RandomForestClassifier(n_estimators=30)
# clf.fit(X_train, y_train)

# Call the function with the specified number of units and the trained classifier
collect_and_save_data(units_in_data)
