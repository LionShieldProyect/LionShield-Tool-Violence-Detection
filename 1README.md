# LionShield - Advanced AI-Powered Violence Detection System

## Revolutionary Technology
LionShield represents a breakthrough in real-time violence detection, combining state-of-the-art OpenPose AI with advanced machine learning algorithms. This innovative fusion creates a system capable of detecting potentially violent actions with remarkable accuracy and speed.

## Core Technology Integration

### 1. OpenPose AI Integration
The system leverages OpenPose AI for precise real-time body keypoint tracking:
- Tracks 7 critical anatomical points with millisecond precision
- Provides continuous skeletal tracking at camera frame rate
- Ensures accurate movement pattern analysis
- Enables real-time pose estimation

### 2. Machine Learning Engine
Our custom-trained machine learning model (`trained_classifier_model_v2.pkl`) delivers:
- Advanced pattern recognition algorithms
- Real-time sequence analysis
- Dual-action probability assessment
- Adaptive confidence scoring

## System Architecture

### 1. Data Processing Pipeline
1. **Input Processing**
   - Real-time video capture
   - Frame-by-frame analysis
   - Keypoint extraction
   - Sequence compilation

2. **Analysis Engine**
   - Pattern matching against trained data
   - Confidence score calculation
   - Action classification
   - Real-time decision making

3. **Output System**
   - Visual feedback generation
   - Confidence percentage display
   - Action classification output
   - System status indicators

## Technical Requirements

### Hardware Requirements
- Computer with camera
- Sufficient processing power for real-time analysis
- Adequate lighting conditions

### Software Components
- Python 3.8+
- OpenCV
- NumPy
- OpenPose model (`graph_opt.pb`)
- Trained classifier model (`trained_classifier_model_v2.pkl`)

## Step-by-Step Installation and Setup Guide

### Step 1: Download and Prepare Files
1. Download all these files to your computer:
   - `action_classifier.py` (the brain of the system)
   - `Bully_detection.py` (the main program)
   - `train_classifier.py` (for training)
   - `graph_opt.pb` (the OpenPose AI model)
   - `trained_classifier_model_v2.pkl` (the trained model)

2. Create these folders in your project directory:
   ```
   training_data/
   ├── neutral/    (for normal action videos)
   └── violence/   (for violent action videos)
   ```

### Step 2: Install Python and Required Packages
1. If you don't have Python installed:
   - Go to python.org
   - Download Python 3.8 or newer
   - Run the installer
   - **Important**: Check "Add Python to PATH" during installation

2. Open Terminal (Mac) or Command Prompt (Windows):
   - On Mac: Press Cmd + Space, type "Terminal", press Enter
   - On Windows: Press Windows + R, type "cmd", press Enter

3. Navigate to your project folder:
   ```bash
   # On Mac:
   cd /path/to/your/project

   # On Windows:
   cd C:\path\to\your\project
   ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

The training of the model is optional because the model that you downloaded, "trained_classifier_model_v2.pkl", has already been trained.
So if you want to skip this, go straight to step 5.

### Step 3: Prepare Training Videos
1. For normal actions (`training_data/neutral/`):
   - Record or collect videos of normal movements
   - Examples: walking, talking, sitting
   - Save as .mov or .mp4 files
   - Put them in the `neutral` folder

2. For violent actions (`training_data/violence/`):
   - Record or collect videos of violent movements
   - Examples: hitting, pushing, aggressive gestures
   - Save as .mov or .mp4 files
   - Put them in the `violence` folder

### Step 4: Train the Model
1. Open Terminal/Command Prompt
2. Navigate to your project folder
3. Run the training script:
   ```bash
   python train_classifier.py
   ```
4. Wait for training to complete
   - You'll see progress messages
   - Training might take several minutes
   - Don't close the terminal while training

### Step 5: Run the Detection System
1. Open Terminal/Command Prompt
2. Navigate to your project folder
3. Run the main program:
   ```bash
   python Bully_detection.py
   ```
4. Position yourself for the camera:
   - Stand 2-3 meters from the camera
   - Make sure your full body is visible
   - Ensure good lighting
   - Stay within camera view

### Step 6: Using the System
1. When the program starts:
   - You'll see your camera feed
   - Green dots will appear on your body
   - A black box will show detection results

2. Understanding the display:
   - Green text = normal movement
   - Red text = violent movement
   - Numbers show confidence percentage

3. To stop the program:
   - Press 'q' on your keyboard
   - Or close the camera window

## Performance Optimization

### Optimal Setup Conditions
- Full body visibility in frame
- Adequate lighting
- Stable camera position
- Clear line of sight

### System Calibration
- Camera positioning
- Lighting adjustment
- Frame rate optimization
- Detection threshold tuning

## Machine Learning Model Architecture

### 1. Training Process
- 60-frame sequence processing
- Pattern recognition training
- Movement classification
- Confidence threshold establishment

### 2. Classification Algorithm
- Distance-based pattern matching
- Weighted confidence scoring
- Dual-action probability assessment
- Adaptive threshold adjustment

### 3. Model Features
- Real-time processing capability
- Adaptive learning
- Confidence-based decision making
- Retraining capability

## Current Capabilities and Future Development

### Current Features
- Single-person detection
- Real-time processing
- Dual-action classification
- Confidence scoring

### Planned Enhancements
- Multi-person detection
- Enhanced accuracy
- Extended action types

## Troubleshooting Common Issues
1. If camera doesn't open:
   - Check if another program is using the camera
   - Try restarting your computer
   - Verify camera permissions

2. If detection isn't working:
   - Check lighting conditions
   - Make sure your full body is visible
   - Verify all files are in the correct locations

3. If you get Python errors:
   - Make sure Python is installed correctly
   - Verify all packages are installed
   - Check if you're in the correct directory

## Technical Support
For technical assistance:
- Email: sdevilla13@icloud.com
- Response time: Within 24 hours

## Project Credits
Development by Sebastián De Villasante using OpenPose for AI-based pose detection

## License
This project is licensed under the MIT License - see the LICENSE file for details.
