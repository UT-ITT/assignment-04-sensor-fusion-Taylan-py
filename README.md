[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AktWbCri)
# assignment-04-CV-Sensor-Fusion

This repository contains the solutions for Assignment 4: Markers, Computer Vision and Sensor Fusion.

## 1. Perspective Transformation
The `image_extractor.py` script located in the `perspective_transformation` directory allows users to select four corner points on an image and warps the selected region into a rectangle of a specified resolution.

### Setup and Requirements
To install the necessary dependencies, run the following command in the root of the repository:
```bash
pip install -r requirements.txt
```

### Usage
Run the script from the `perspective_transformation` directory:
```bash
cd perspective_transformation
python image_extractor.py <input_image> <output_image> <resolution>
```

**Example:**
```bash
python image_extractor.py sample_image.jpg result.jpg 800x600
```

### Controls
- **Left Click**: Select 4 corner points on the image.
- **ESC**: Clear selected points and start over.
- **S**: Save the warped image to the specified output path (after 4 points are selected).
- **Q**: Quit the application.

## 2. AR Game (Target Destroyer)
The `AR_game.py` script located in the `ar_game` directory is an Augmented Reality mini-game built using OpenCV and Pyglet. 

It reads from your webcam, detects a board with 4 ArUco markers, warps the board into the game screen, and lets you play "Target Destroyer" by tracking a **blue object** (like a blue pen cap, blue post-it, or even your phone screen displaying the color blue) to smash targets!

### Usage
Run the script from the `ar_game` directory:
```bash
cd ar_game
python AR_game.py
```
If you don't have a webcam and want to test it with a pre-recorded video:
```bash
python AR_game.py path/to/your_test_video.mp4
```

### How to Play
1. Point your camera at a surface with 4 ArUco markers. The game will automatically detect them and warp the area between them to fill your window.
2. Hold a **distinctly BLUE object** in front of the camera inside the warped area. A green "Player" crosshair will appear over it.
3. Move the blue object to hover over the red bullseye targets to destroy them and increase your score!
5. **Controls**: Press `ESC` or `Q` to quit at any time.

## 3. Sensor Fusion
The `sensor_fusion.py` script located in the `sensor_fusion` directory fuses absolute camera-based tracking (using ArUco markers) with high-frequency relative tracking (using a smartphone's accelerometer via DIPPID) through a **Complementary Filter**.

### Usage
Run the script from the `sensor_fusion` directory:
```bash
cd sensor_fusion
python sensor_fusion.py [DIPPID_PORT]
```
*(The `DIPPID_PORT` is optional and defaults to `5700`)*

### How to Test
1. Start the script and show the camera a board of 4 ArUco markers to establish the tracking region.
2. Open the **DIPPID app** on your smartphone and ensure it is broadcasting accelerometer data to port `5700`.
3. Display an **ArUco marker** (e.g., ID 5 or 23) on your smartphone's screen, or tape a printed marker to its back.
4. Move your smartphone around within the board area:
   - **Red Dot**: Raw Camera Position.
   - **Green Dot**: Predicted Position (fusing camera and accelerometer data).
5. **Adjust Alpha Weight**: Use the `Left` and `Right` arrow keys to change the Complementary Filter's `alpha` parameter in real-time.
   - *High Alpha (~0.9)*: Relies heavily on the accelerometer. Highly responsive but prone to drift over time due to integration error.
   - *Low Alpha (~0.1)*: Relies heavily on the camera. Absolute and stable, but can lag slightly.
6. **Reset Prediction**: Press **Button 1** on the DIPPID app to instantly snap the predicted (green) position back to the camera's raw (red) position.