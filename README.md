# Faster RCNN for Traffic Sign Detection

This repo is half of the course project for COMP7502 Image Processing and Computer Vision at The University of Hong Kong on traffic sign detection.

# IMPORTANT NOTE

This repo will be **removed** from GitHub anytime on or after January 1, 2018. (UTC+8)

Git LFS currently does not support removing files still referenced by old commits, so keeping this repo on GitHub will require a monthly fee unless we remove existing commits (which is undesirable).

As the course project has finished grading, maintaining this repo is no longer sensible.

If you would like to continue using this repo, please clone it before January 1, 2018. (UTC+8)

## Installation

This repo is self-contained. You can clone and run this repo by following instructions in readme.pdf.

If you have the submitted version of this repo, which has all the dataset and model files removed, determine and download the files you need:

 1. All files under the `dataset` directory are needed
 2. The `GTSRB/models/gtsrb1/gtsrb1-last.hdf5` model file is needed if you want to run the GTSRB classifier using the trained model. Other model files in that directory is for record-keeping only
 3. The `GTSRB/tblogs/gtsrb1/events.out.tfevents.1501183018.twin-VirtualBox` event file is needed if you want to run Tensorboard visualization of the GTSRB classifier training
 4. The `keras-frcnn/vgg16_weights_tf_dim_ordering_tf_kernels.h5` model file is needed if you want to train a FRCNN model from pretrained VGG16 using the `keras-frcnn` code.
 5. The `keras-frcnn/simple_model_frcnn.hdf5` model file is needed if you want to train/run our trained model of our simplified FRCNN network.
 
## License

All files under the `GTSRB` and `GTSDB` directories are lincensed under the MIT License. See `LICENSE` for more information.

Files under the `keras-frcnn` directory was derived from the GitHub repo [yhenon/keras-frcnn](https://github.com/yhenon/keras-frcnn), which is licensed under the Apache 2.0 License. See that repo for more information.

The dataset directory contains preprocessed versions of the GTSRB and GTSDB datasets freely available at http://benchmark.ini.rub.de/?section=home&subsection=news.
