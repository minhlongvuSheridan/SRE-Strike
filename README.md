<h1 align="center">👓 RSE Strike 👓 </h1>

# Description 📝
RSE Strike is a simple AR game that allows users to attack other users with fire effects. RSE stands for Recognition, Segmentation, and Estimation, respectively. Each represents a major functionality of the application.

- ***Segmentation***: Instance segmentation is used to determine the exact bounded area of the face, not just a bounding box. This acts as the hitbox.

- ***Recognition***: This uses **Face Recognition** to identify the user's face. It uses **FaceNet** to extract embeddings and a **Support Vector Machine (SVM)** to classify the image.

- ***Estimation***: Hand pose estimation is used to determine the points corresponding to fingers on a hand. These points represent hand gestures, which determine whether to trigger or embed visual effects.
