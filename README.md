# DD-Net: Dual decoder network with curriculum learning for full waveform inversion

## Abstract
Deep learning full waveform inversion (DL-FWI) is gaining much research interests due to its high prediction efficiency, effective exploitation of spatial correlation, and not requiring any initial estimate. As a data-driven approach, it has several key issues, such as: how to design effective deep models, how to control the training process, and how to enhance the generalization ability. In this paper, we propose a dual decoder network with curriculum learning (DD-Net) to handle these issues. First, we design a U-Net with two decoders to grasp the velocity value and stratigraphic boundary information of the velocity model. These decoders' feedback will be combined at the encoder to enhance the encoding of edge spatial information. Second, we introduce curriculum learning to model training by organizing data in three difficulty levels. The easy-to-hard training process enhances the data adaptability of the model. Third, we generalize the model to new environments via a pre-network dimension reducer. In this way, the prediction performance is enhanced on data from different fields. Experiments were undertaken on SEG salt datasets and four synthetic datasets from OpenFWI. Results show that our model is superior to other state-of-the-art data-driven models.

![image](DDNet.png)

## Folder: results
Store intermediate and final results of model runs.
These results include drawing loss curves and loss arrays saved with .mat.
The .npy file saves the evaluation results of the corresponding model for each test data.

## Folder: results
