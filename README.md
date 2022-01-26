## Setup environment

You can change the pytorch, torchvision version (in build_environment.sh) to fit your GPU

    source build_environment.sh $ENV_NAME

## Data preprocessing (TEyeD dataset)

Put TEyeD dataset directory at $PATH. The preprocessed data will be under $PATH/TEyeD

    python3 Dikablis_preprocess.py --root $PATH

## Data preporcessing (Neurobit dataset)

Please refer to the directory tree in Neurobit_data.py

    python3 Neurobit_data.py --root $ROOT --data_dir $DATA_PATH
    
## Train gaze

    python train_gaze.py --data_dir $PATH/TEyeD --dataset TEyeD

## Train validity

    python train_gaze.py --data_dir $PATH/TEyeD --dataset TEyeD

## Inference

    python gaze_visualization.py --output_video $OUTPUT_PATH

## Citation
* https://arxiv.org/pdf/1807.10002.pdf
* https://arxiv.org/abs/2102.02115
* https://github.com/milesial/Pytorch-UNet
* https://github.com/princeton-vl/pytorch_stacked_hourglass
* Appearance-based Gaze Estimation in the Wild, X. Zhang, Y. Sugano, M. Fritz and A. Bulling, Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June, p.4511-4520, (2015). 
* @inproceedings{zhang15_cvpr,
  Author = {Xucong Zhang and Yusuke Sugano and Mario Fritz and Bulling, Andreas},
  Title = {Appearance-based Gaze Estimation in the Wild},
  Booktitle = {Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  Year = {2015},
  Month = {June}
  Pages = {4511-4520} }
* @article{ICML2021DS,
  title={TEyeD: Over 20 million real-world eye images with Pupil, Eyelid, and Iris 2D and 3D Segmentations, 2D and 3D Landmarks, 3D Eyeball, Gaze Vector, and Eye Movement Types},
  author={Fuhl, Wolfgang and Kasneci, Gjergji and Kasneci, Enkelejda},
  journal={arXiv preprint arXiv:2102.02115},
  year={2021}
  }
