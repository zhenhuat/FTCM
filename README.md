# FTCM
This is the readme file for the code release of "(TCSVT 2023) FTCM: Frequency-Temporal Collaborative Module for Efficient 3D Human Pose Estimation in Video" on PyTorch platform.

Thank you for your interest, the code and checkpoints are being updated.



## The released codes include:
    checkpoint/:                        the folder for model weights of FTCM.
    dataset/:                           the folder for data loader.
    common/:                            the folder for basic functions.
    model/:                             the folder for FTCM network.
    run_ftcm.py:                         the python code for FTCM networks training.


## Dependencies
Make sure you have the following dependencies installed:
* PyTorch >= 0.4.0
* NumPy
* Matplotlib=3.1.0

## Dataset

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

### Human3.6M
We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). 
### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset in the same way as [P-STMO](https://github.com/paTRICK-swk/P-STMO). 


## Training from scratch
### Human 3.6M
For the training stage, please run:
```bash
python run_stc.py -f 243 -b 512  --train 1 --layers 6 
```
For the testing stage, please run:
```bash
python run_stc.py -f 243 -b 512  --train 0 --layers 6 --reload 1 --previous_dir ./checkpoint/your_best_model.pth
```


## Evaluating our models

You can download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1Am_SJ9cUh9xDO7tdC0sF8Pl6x4kg2TrR?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1ga_UDRy1eK9cBDTYCALdkA) (extraction code：FTCM). Put them in the ./checkpoint directory.

### Human 3.6M

To evaluate our FTCM model with refine module on the 2D keypoints obtained by CPN, please run:

```bash
python run_myftcm_exp.py --train 0 --reload 1 -tds 3 --f 243 --previous_dir ./checkpoint/model_243_refine/no_refine_6_4331.pth --refine --refine_reload 1 --previous_refine_name ./checkpoint/model_351_refine/refine_6_4331.pth
```

Different models use different configurations as follows.

| Input | Frames | P1 (mm) | P2 (mm) | 
| -------------| ------------- | ------------- | ------------- |
| CPN | 243  | 43.32  | 34.92  |



### MPI-INF-3DHP
The pre-trained models and codes for STCFormer are currently undergoing updates. 


## Citation

If you find this repo useful, please consider citing our papers:

@article{tang2023ftcm,
  title={FTCM: Frequency-Temporal Collaborative Module for Efficient 3D Human Pose Estimation in Video},
  author={Tang, Zhenhua and Hao, Yanbin and Li, Jia and Hong, Richang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}

and

@inproceedings{tang20233d,\
  title={3D Human Pose Estimation With Spatio-Temporal Criss-Cross Attention},\
  author={Tang, Zhenhua and Qiu, Zhaofan and Hao, Yanbin and Hong, Richang and Yao, Ting},\
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\
  pages={4790--4799},\
  year={2023}
}

## Acknowledgement
Our code refers to the following repositories.

[VideoPose3D](https://github.com/facebookresearch/VideoPose3D) \
[StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D) \
[P-STMO](https://github.com/paTRICK-swk/P-STMO/tree/main) \
[STCFormer](https://github.com/zhenhuat/STCFormer) 

We thank the authors for releasing their codes.
