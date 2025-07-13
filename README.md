Due to loss of server access permissions, partial reference code is provided. The implementation of EdgeCAM is based on the **trochcam** library, and only minor modifications to **Grad-CAM++** are required to complete the computation . The project is not an end-to-end structure, and the training steps are as follows:  
1) Train a tampering detection binary classification model, saving weights for each epoch (which consumes significant storage space);  
2) Calculate candidate SOP (already provided);  
3) Compute EdgeCAM visualizations using models from different training stages;  
4) Fuse EdgeCAM results to obtain coarse tampered region edges;  
5) Use EdgeCAM edges to select the final tampered region mask from candidate SOP.  
For continuous assistance, contact **yzhoulv@foxmail.com**.  



SOP：https://pan.baidu.com/s/1PXUHecO7TT66DUikOh4epg?pwd=vfms
code：vfms


ps-battles dateset:
https://drive.google.com/file/d/10E5Trc3r_ScE4VeyrZ0tGLKM7HTCu3xt/view?usp=drive_link
