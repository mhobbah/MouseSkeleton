# Mouse Skeleton

## Introduction
This repository contains most of the code used for the Master's Thesis: 3D Reconstruction from Multi-View video Data with Advanced Machine Learning and Optimization Techniques. This thesis stems from the collaboration of the Modelization of Cognitive Processes and the Computer Vision Labs from TU Berlin. This work has been supervised by Mrs. Soledad Traverso.

## Description
The goal of this thesis was to develop a method to consitently reconstruct the skeleton of a moving mouse in a lockbox experiment. Despite several cameras are employed to record the mouse, occlusion of the mouse's body parts is still a problem. To solve this, we simulate a moving mouse (tracks avalaible in **`glob3DBodyParts.csv`**) and we introduce three different types of masking that we also call modes (*Coordinates*, *Parts* and *Frames*). Their exhaustiv description is written in the Methodology chapter of the thesis and their implementation is accessible in **`dataset.py`**. Then, we compare different networks on the reconstruction task by training them on the simulated tracks and evaluating them on three test sets that come from the original simulated tracks but with some variation (*seed*, *speed* and *angle). Their description is also available in the thesis but in the Evaluation chapter. To further evaluate the models' performance, we visualize tracks and skeletons in **`Visualization.ipynb`**. Finally, visualization of the predicted moving skeleton is also possible by computing a csv file with **`tracks_csv.py`** and using a C++ script.

## Files
- **`models\`**: contains the trained models
- **`networks\`**: contains the implementation of the networks
- **`results\`**: contains the evaluation of the modelson the metrics
- **`test_sets\`**: contains the test sets on which the models are evaluated
- **`dataset.py`**: loads and preprocesses data, the artificial masking is part of the preprocessing step
- **`glob3DBodyParts.csv`**: the simulated mouse's tracks, file contain rigid and deformable coordinates over time
- **`main.py`**: runs the training of a network and its evaluation on the test sets
- **`metrics.py`**: runs the evaluation of a network on the test sets for one type of masking
- **`Stats.ipynb`**: presents statistics of the dataset
- **`tools.py`**: various helpful functions
- **`tracks_csv.py`**: saves the tracks computed by a trained model in a csv file in the same format as **`glob3DBodyParts.csv`**
- **`training.py`**: runs the training of a network for one type of masking
- **`Visualization.ipynb`**: presents visualization of tracks and skeletons computed by a trained model


## Installation and execution
### Instructions
1. Clone this repository:
   
   ```bash
   git clone https://github.com/mhobbah/mouse_skeleton.git
   ```
2. Install depedencies:
   
   ```bash
   pip install -r requirements.txt
   ```
3. Execute main script:
   Typical use case (example):
   ```bash
   python main.py --model_type AGCN --key symmetry
   ```

   Test run (example):
   ```bash
   python main.py --model_type AGCN --key symmetry --par_dir ./tests --filename agcnsymtest --test True
   ```
4. Execute other scripts:
   Train one mode (example):
   ```bash
   python training.py --model_type AGCN --key symmetry --mode 0
   ```

   Evaluate one mode (example):
   ```bash
   python metrics.py --model_type AGCN --key symmetry --mode 0 --filename agcnsym0
   ```

   Compute 3D tracks (example):
   ```bash
   python tracks_csv.py --model_type AGCN --key symmetry --mode 0 --filename agcnsym0
   ```