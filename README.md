# AAT: Attribute-prompted Adapter Tuning for Efficient Video Action Recognition

<p align="center"><img src="./figure/actions_storage.jpg" width="100%" alt="" /></p>
<p align="center"><img src="./figure/distribution_UMAP.png" width="100%" alt="" /></p>

## Installation

- Create a conda environment: ```conda create -n aat python=3.9```
- Install PyTorch 1.7.1+ and Torchvision 0.8.2+ 
- ```pip3 install -r requirements.txt```

## Dataset setup

```bash
${ROOT}/
|-- dataset
|   |-- CAD
|   |   |-- data_train_CAD.npz
|   |   |-- data_test_CAD.npz
|   |-- MCPRL
|   |   |-- data_train_MCPRL.npz
|   |   |-- data_test_MCPRL.npz
```

## Download pretrained model

The pretrained model can be found in [here](), please download it and put it in the ```'./checkpoint/pretrained'``` directory of the corresponding baseline methods.  

## Test the model

To test on a pretrained model on MCPRL:  

```bash
python main.py --test --previous_dir 'checkpoint/pretrained/'
```

## Train the model

To train on a model on MCPRL:

```bash
python main.py --batch_size 1024
```



## Citation

If you find our work useful in your research, please consider citing:

## Licence

This project is licensed under the terms of the MIT license.
