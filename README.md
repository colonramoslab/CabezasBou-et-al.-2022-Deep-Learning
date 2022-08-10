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

2 SENTENCE EXPLANATION OF PROJECT. For this project, C. elegans behavior and speed was tracked using the below described code and techniques. The code is available for installation on all major platforms (e.g. Windows, Linux, OS X) and GitHub. 

# System Requirements and Installation Guide

## Hardware Requirements

Only a standard computer with enough RAM to support the in-memory operations is necessary to support the in-memory operations.

## Software Requirements

To install MATLAB - https://www.mathworks.com/products/matlab.html

To install DeepLabCut (version 2.2b8 used) - https://github.com/DeepLabCut/DeepLabCut

To install CytoSHOW - http://www.cytoshow.org/

# Neuron Tracking and Data Visualization

Calcium imaging data in freely moving worms performing thermotaxis was recorded and analyzed in CytoShow. Data was aligned manually with the channels offset to correct for pixel overlay. 


# Image Processing



# Calcium Activity Deep Learning Analysis


# Behavior Deep Learning Analysis

To generate a clear, robust video of the worm for tracking, a script (FIJI_Image_Macros/FIJI_Merge_Channels_Script_GCAMP6_RFP.txt) was used to overlay dual channel recordings of worms expressing GCAMP6 and RFP in Fiji. First, the GCAMP6 and RFP videos were separately thresholded (GCAMP6 = 94,127, RFP = 90,125). Then, both sets of videos had their backgrounds subtracted (rolling-30 stack option). Then, the channels were merged using “Merge Channels”. Lastly, the merged videos were saved in ‘.AVI’ format with no compression and in the default 7 frames per second.

Next, a convoluted neural network (CNN) was trained using DeepLabCut (Mathis et al., 2018; Nath et al., 2019) and used as a marker-less pose estimator to track _C. elegans_ behavior. Implementation of this approach was performed as previously described in Hawk et al., 2021 with key modifications as described in the paper. The associated config folder can be found here (config.yaml).


# Citations

Hawk JD, et al. Integration of Plasticity Mechanisms within a Single Sensory Neuron of C. elegans Actuates a Memory. Neuron 97, 356-367 e354 (2018). 

Gershow M, et al. Controlling airborne cues to study small animal navigation. Nat Methods 9, 290-296 (2012).

Luo L, et al. Bidirectional thermotaxis in Caenorhabditis elegans is mediated by distinct sensorimotor strategies driven by the AFD thermosensory neurons. Proc Natl Acad Sci U S A 111, 2776-2781 (2014).

Mathis A, et al.DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci21, 1281-1289 (2018).

Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc14, 2152-2176 (2019).

# License

https://github.com/colonramoslab/Hawk-et-al.-HySyn-2021/blob/beb04a038bfa8ba9a3e0cdb68b20a6ac855df303/LICENSE
