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

Through the development of a novel, live-imaging system that simultaneously tracks calcium levels and worm behavior, we were able to see how, in C. elegans, sensory neuronal activity correlates with sensory information processing and the navigation strategy that produces goal-directed behavior. Thus, this approach not only offers insight into these questions, but promises to shed further light on the more general processes underlying the acquisition of behavioral preferences and the role played by AFD therein. For this project, analysis was accomplished with the described code and techniques with the purpose of tracking different facets of worm locomotion. The code is available for installation on all major platforms (e.g. Windows, Linux, OS X) and GitHub. 

# System Requirements and Installation Guide

## Hardware Requirements

Only a standard computer with enough RAM to support the in-memory operations is necessary to support the in-memory operations.

## Software Requirements

To install MATLAB - https://www.mathworks.com/products/matlab.html

To install DeepLabCut (version 2.2b8 used) - https://github.com/DeepLabCut/DeepLabCut

To install CytoSHOW - http://www.cytoshow.org/

To install BeanShell - https://beanshell.github.io/

To install ImageJ/Fiji - https://imagej.nih.gov/ij/

# Neuron Tracking and Data Visualization

To reliably record neuronal calcium activity as the worms freely navigated the thermal gradient, we developed a neuron tracking system coded in Beanshell script (CabezasBou-et-al.-2022/Neuro_Tracker_Codes/).  

# Image Processing

Calcium imaging data in freely moving worms performing thermotaxis was recorded and analyzed in CytoShow. Data was aligned manually with the channels offset to correct for pixel overlay. Channels were split using an ImageJ script (FIJI_Image_Macros/FIJI_Split_Channels_GCaMP_RFPijm) with the purpose of individually detecting RFP and GCAMP6 fluorescent neurons.

# Calcium Activity Deep Learning Analysis

A fully convolution network (FCN) was used to predict a binary AFD skeleton mask for every individual green and red fluorescent channel image , which was then used to precisely quantify the location and pixel intensity of the GCaMP and RFP in all parts of the neuron. Since the model was trained to detect AFD’s soma as separate from the dendrite structure, our analysis was able to quantify neuronal activity with subcellular resolution.

# Behavior Deep Learning Analysis

To generate a clear, robust video of the worm for tracking, a script (FIJI_Image_Macros/FIJI_Merge_Channels_Script_GCAMP6_RFP.txt) was used to overlay dual channel recordings of worms expressing GCAMP6 and RFP in Fiji. 

Next, a convoluted neural network (CNN) was trained using DeepLabCut (Mathis et al., 2018; Nath et al., 2019) and used as a marker-less pose estimator to track _C. elegans_ behavior. Implementation of this approach was performed as previously described in Hawk et al., 2021 with key modifications as described in the paper. Data from DLC was analyzed using a Jupyter script that can be found here (Freely_Moving_Analysis_Code/Freely Moving Analysis Example Code/Full_Cal_Model-Copy1(3_4_20_17to24_1_2_).ipynb). The associated config folder can be found here (config.yaml).

# Citations

Hawk, J. D., Wisdom, E. M., Sengupta, T., Kashlan, Z. D., &amp; Colón-Ramos, D. A. (2021). A genetically encoded tool for reconstituting synthetic modulatory neurotransmission and reconnect neural circuits in vivo. Nature Communications, 12(1). 

Mathis A, et al.DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci21, 1281-1289 (2018).

Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc14, 2152-2176 (2019).

# License

https://github.com/colonramoslab/CabezasBou-et-al.-2022/blob/main/LICENSE
