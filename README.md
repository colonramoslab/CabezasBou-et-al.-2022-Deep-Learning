# Deep Learning approaches to understanding sensory neuron activity and goal-directed behaviors in freely moving animals

## Contents

- [Overview](#overview)
- [System Requirements and Installation Guide](#system-requirements-and-installation-guide)
- [Neuron Tracking and Data Visualization](#Neuron-Tracking-and-Data-Visualization)
- [Image Processing](#Image-Processing)
- [Calcium Activity Deep Learning Analysis](#Calcium-Activity-Deep-Learning-Analysis)
- [Behavior Deep Learning Analysis](#Behavior-Deep-Learning-Analysis)
- [Citations](#Citations)
- [License](#license)

# Overview

HySyN is a system designed to rewire neural connectivity in vivo by reconstituting synthetic modulatory neurotransmission. We demonstrate that genetically targeted expression of the two HySyn components, a Hydra-derived neuropeptide and its receptor, createsde novoneuromodulatory transmission in a mammalian neuronal tissue culture  model  and  rewires  a  behavioral  circuit in  vivoin  the  nematode Caenorhabditis elegans. For this project, C. elegans behavior and speed was tracked using the below described code adn techniques. The code is available for installation on all major platforms (e.g. Windows, Linux, OS X) and GitHub. 

# System Requirements and Installation Guide

## Hardware Requirements

Only a standard computer with enough RAM to support the in-memory operations is necessary to support the in-memory operations.

## Software Requirements

To install MATLAB - https://www.mathworks.com/products/matlab.html

To install DeepLabCut (version 2.2b8 used) - https://github.com/DeepLabCut/DeepLabCut

# Neuron Tracking and Data Visualization

The purpose of this code was to monitor C. elegans migration on an agar pad over time, and calculate the mean speed over time as well as distanced traveled. Methods overview: Synchronized young adult nematode populations were washed in M9 buffer then transferred by pipette to the 20°C behavioral test plate (22-cm x 22-cm agar plates). Worms were obliquely illuminated using an array of 624nm LEDs and migration was monitored for 120 min at 2fps using a MightEx camera (BCE-B050-U). Animal speed was analyzed using an adaptation of the MagatAnalyzer software package (Gershow et al. 2012, Luo et al. 2014) and custom MATLAB scripts (posted to this repository). Briefly, MagatAnalyzer uses a published and well-characterized machine vision approach to extract animal centroid position over time (Hawk et al. 2018). To estimate speed over the entire trajectory, we calculated the slope of a linear fit of the displacement for this centroid over time.


# Image Processing

The purpose of this code was to analyze head thrashing movement of the C. elegans by quantifying the radial velocity of the head and tail as well as thrashing, which was estimated by calculating the head angle vector. Methods overview: Worm behavior was tracked using DeepLabCut (version 2.2b8), a deep convolution network that utilizes pretrained residual networks to robustly track animal behavior (Mathis et al. 2018). We labeled 10 videos using DeepLabCut’s GUI toolbox with the following changes todefault config file parameters: 8 evenly-spaced body parts from head to tail were labeled across 15 frames (set as ‘numframes2pick’value); label size was set to 2, alphavalue to 0.7, p-cutoff to 0.9.  Frames  to  be  labeled  were  extracted  using  the  k-means  clustering  function.  Other  default parameters for extracting frames were maintained. Following thisstep, the frames were labeledmanually,  a  training  set  was  created  using  the  resnet_50  network  and  default  augmentation method,  and  the  network  was  trained  with  the  default  parameters  outlined  in prior  work (Nath, Mathis A, Patel, Bethge, Mathis MW, 2019). The resulting network was used to label all frames across each video and body part locations in each frame  were  saved  for  downstream  analyses. To  estimate  thrashing,  the  head  angle  vector, described  by  the first two  points  along the  body axis  in  the  head  and the  neck,  was  quantified across  all  frames.  After  aligning  to  a  common  origin,  the  change  in  angle  of  this  vector  was calculated between successive frames captured at 30 fps. The median headangle change per second value was used to quantify individual animal thrashing. Illustration of angle quantification is available in supplemental videosSV2 (full speed) and SV3 (10x slower speed clip).

# Calcium Activity Deep Learning Analysis

# Behavior Deep Learning Analysis

# Citations

Hawk JD, et al. Integration of Plasticity Mechanisms within a Single Sensory Neuron of C. elegans Actuates a Memory. Neuron 97, 356-367 e354 (2018). 

Gershow M, et al. Controlling airborne cues to study small animal navigation. Nat Methods 9, 290-296 (2012).

Luo L, et al. Bidirectional thermotaxis in Caenorhabditis elegans is mediated by distinct sensorimotor strategies driven by the AFD thermosensory neurons. Proc Natl Acad Sci U S A 111, 2776-2781 (2014).

Mathis A, et al.DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci21, 1281-1289 (2018).

Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc14, 2152-2176 (2019).

# License

https://github.com/colonramoslab/Hawk-et-al.-HySyn-2021/blob/beb04a038bfa8ba9a3e0cdb68b20a6ac855df303/LICENSE
