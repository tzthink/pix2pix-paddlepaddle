Introduction
============
A PaddlePaddle implementation for [pix2pix](https://phillipi.github.io/pix2pix/) will be given. Facade dataset is used for the training and test. 


### Reference
* [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)</li>
* [TensorFlow implementation for pix2pix](https://github.com/affinelayer/pix2pix-tensorflow)</li>


Prerequisites
=============

### Hardware and software requirements
```
Nvidia GPU,
cuda                9.0,
cudnn               7.1.4,
paddlepaddle-gpu    0.14.0.post87
```

### Verify a successful installation
We can do the validation by using PaddlePaddle [quick use code](http://www.paddlepaddle.org/docs/0.14.0/documentation/fluid/en/getstarted/quickstart_en.html). We also can switch to use CUDA during the inference via `place=fluid.CUDAPlace(0)`
```
(venv) tzhou@tong-ubuntu:~/PycharmProjects/pix2pix-paddlepaddle$ python quick_start.py 
Predicted price: $12,273.97
```

One common error while setting up paddlepaddle-gpu: `Error: ImportError: libmklml_intel.so: cannot open shared object file: No such file or directory` <br />
The solution is to add `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tzhou/PycharmProjects/pix2pix-paddlepaddle/venv/local/lib` in the `~/.bash_profile`

Usage
=====

### Facades dataset downloading
```
./download_dataset.sh
```

### Training
```
python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 1 \
  --input_dir facades/train \
  --which_direction BtoA
  ```
  
  
Results
=======