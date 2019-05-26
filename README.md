# SF-ID-Network-For-NLU

[![License](https://camo.githubusercontent.com/8051e9938a1ab39cf002818dfceb6b6092f34d68/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d417061636865253230322e302d626c75652e737667)](https://opensource.org/licenses/Apache-2.0) 

This is the source implementation of ACL2019 accepted paper: A Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot Filling.

## Model Notes

![Alt text](https://github.com/ChenZhongFu/SF-ID-Network-For-NLU/blob/master/docs/framework.png)

**Our model🚀🚀🚀**

The SF-ID network consists of an SF subnet and an ID subnet. The order of the SF and ID subnets can be customized. Depending on the order of the two subnets, the model have two modes: SF-First and ID-First. The former subnet can produce active effects to the latter one by a medium vector.

## Installation

For training, a GPU is strongly recommended for speed. CPU is supported but training could be extremely slow.

### Tensorflow

The code is based on Tensorflow and supports **Tensorflow-gpu1.11.0** now . You can find installation instructions [here](https://www.tensorflow.org/).

### Dependencies

The code is written in Python 3.5.2. Its dependencies are summarized in the file `requirements.txt`. You can install these dependencies like this:

```
pip3 install -r requirements.txt
```

## Data

We mainly focus on the ATIS dataset and Snip dataset, and the code takes its original format as input. You should be able to get it [here](https://github.com/ChenZhongFu/SF-ID-Network-For-NLU/tree/master/data).

### Format

We assume the corpus is formatted as same as the ATIS dataset and Snips dataset. More specifically, **seq.in**, **label** and **seq.out** is our data files.

The input and output of the training data is as follows:
The input is a sentence in **seq.in** in the format:
```
i want to fly from baltimore to dallas round trip
```
The output is divided into **label output** and **sequence label output**:
The output of the label is in the **label** file in the format:
```
atis_flight
```
The output of the sequence label is in **seq.out** in the format:
```
O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip
```

### Following is the structure and description of out files:

  - **train.py**: The code that is used to train the model containing SF-ID network.
  - **train_ID_only.py**: The code that is used to train the train only containing SF subnet.
  - **train_SF_only.py**: The code that is used to train the train only containing ID subnet.
  - **utils.py**: It contains some functions relevant with data processing and the ev aluation of FI score in slot-filling task.
  - **data**: It involves the data used in the experiments.
    - **atis**: This file is about atis dataset, including train/valid/test set.
    - **snips**: This file is about snips dataset, including train/valid/test set.
    
### Training parameters:

  - **dataset**: It depends on which dataset is used in the experiment, atis or snips.
  - **priority_order**: It depends on the execution order of the two subnets, slot-first or i ntent-first.
  - **use-crf**:  It  depends  on  whether  CRF  layer  is  used,  it  is  belongs  to  boolean  type.  W hen it is set as True, CRF layer is used.
  - **iteration number**: It depends on the iteration number used in iteration mechanism.
  
### Code reference

[SlotGated-SLU](https://github.com/MiuLab/SlotGated-SLU)
  
## Usage

Here we provide implementations for our models, it is **SF-ID-Network**.
A command example is as follow:
```
python train.py --dataset=atis --priority_order=slot_first --use_crf=True
```

## Reference

If you use the code, please cite the following paper: **"A Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot Filling"** Haihong E, Peiqing Niu, Zhongfu Chen, Meina Song. *ACL (2019)*
```
@inproceedings{
  under construction...
}
```

 



