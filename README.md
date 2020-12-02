# Humanoid Soccer Robot Localization
The aim of this project is to localize the robot position in soccer field. I use Augmented Monte Carlo Localization (AMCL) as is basis algorithm. The basic workflow of this code is receiving an input image, extract the features, use the features as observation in AMCL, and do correction in AMCL.

## Features
* Using Intel SSE in projection calculation and pose estimation.
* Enable to reuse by change the feature extraction method.

## Demo
[![Demo](https://img.youtube.com/vi/DUCrRDXByFc/0.jpg)](https://www.youtube.com/watch?v=DUCrRDXByFc)
