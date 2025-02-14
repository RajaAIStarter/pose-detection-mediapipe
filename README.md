# Pose Detection Project

## Overview

This project focuses on pose detection using OpenCV and MediaPipe. A limited set of pose images were taken, and key body landmarks were extracted and stored in CSV files for training and testing. The model was trained using a Random Forest classifier and can be improved further for higher accuracy.

## Key Landmarks Extracted

```
landmark_names = {
    11: 'Left Shoulder',
    12: 'Right Shoulder',
    13: 'Left Elbow',
    14: 'Right Elbow',
    15: 'Left Wrist',
    16: 'Right Wrist',
    23: 'Left Hip',
    24: 'Right Hip',
    25: 'Left Knee',
    26: 'Right Knee',
    27: 'Left Ankle',
    28: 'Right Ankle'
}
```

## Project Structure

```
/pose-detection-project
│── train_pose_landmarks.csv   # Training dataset
│── test_pose_landmarks.csv    # Testing dataset
│── data_extractor.py          # Extracts pose landmarks from real-time feed or dataset
│── model.py                   # Trains Random Forest Classifier and saves model
│── rf_model.pkl               # Trained Random Forest model
│── scaler.pkl                 # Scaler for normalizing real-time data
│── classifier_api.py          # Tests model with images or real-time feed
│── application.py             # Streamlit interface for testing the model
│── DATASET/                   # Contains training & testing images
│── requirements.txt           # Dependencies required for running the project
│── README.md                  # Project documentation
```

## How It Works

1. **Data Extraction**

   - Uses OpenCV and MediaPipe to extract pose key points.
   - Saves extracted landmarks in `train_pose_landmarks.csv` and `test_pose_landmarks.csv`.
   - `data_extractor.py` can be modified to extract data from either live feed or dataset images.

2. **Model Training**

   - `model.py` trains a Random Forest classifier using extracted landmarks.
   - The trained model is saved as `rf_model.pkl`.
   - A scaler (`scaler.pkl`) is used to preprocess live data before classification.
   - Model accuracy can be improved by hyperparameter tuning, increasing dataset size, and using different algorithms.

3. **Testing & Real-Time Detection**

   - `classifier_api.py` allows testing using images or real-time camera feed.
   - Users can specify input sources for testing.
   - The system normalizes live input data before passing it to the model for classification.

4. **User Interface**

   - `application.py` provides a Streamlit-based interactive UI.
   - Allows users to upload images or use webcam for real-time pose detection.
   - Displays detected key points and classified poses in an intuitive way.

## Future Enhancements

- **Expand Dataset**: Add more poses like exercises, yoga, and pilates for better generalization.
- **Improve Accuracy**: For higher accuracy (>99%), use CNN-based deep learning models instead of Random Forest.
- **Edge Device Optimization**: Since it uses MediaPipe, it can run efficiently on edge devices like Raspberry Pi.
- **Additional Features**: Implement custom pose classification, gesture recognition, and integration with fitness applications.
- **Real-Time Performance Enhancements**: Optimize real-time detection pipeline to reduce latency and improve FPS.

## Installation & Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/pose-detection.git
   cd pose-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Extract pose landmarks:

   ```bash
   python data_extractor.py
   ```

4. Train the model:

   ```bash
   python model.py
   ```

5. Test the model:

   ```bash
   python classifier_api.py
   ```

6. Run the application:

   ```bash
   streamlit run application.py
   ```

## Conclusion

This project provides a base for pose detection using MediaPipe and machine learning. Further improvements can be made to enhance accuracy and add more features. The framework is designed to be flexible, allowing for easy modifications and integrations into other applications.

