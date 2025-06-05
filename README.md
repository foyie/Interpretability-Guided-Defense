# Interpretability-Guided-Defense

* This is the repository of the Interpretability-Guided Test-Time Adversarial Defense
    * This work focuses on neuron importance ranking with the aim of improving adversarial robustness.
    * This is first of its kind to be training free.
    * The work was evaluated on CIFAR10, CIFAR100 and ImageNet-1k
    * The baselines models are FAT, GAT, NuAT, TRADES, AWP, and others.
    * We worked towards reproducing the project on the [github page](https://github.com/Trustworthy-ML-Lab/Interpretability-Guided-Defense/tree/main).


* Team Members - Shweta Nalluri, Chandrima Das, Rohan Thorat, Kavit Shah    








<p align="center">
<img src="https://github.com/user-attachments/assets/5cd73bf7-c8c7-4707-8828-a6be5ad21c64" width="900">
</p>

## Requirements

The configuration we worked on was 8 CPU, 16G RAM, 1 GPU.

We installed the required packages using the following commands:

pip install -r requirements.txt
pip install git+https://github.com/RobustBench/robustbench.git


### Pretrained weights

* Downloaded the models through the following links:
    * [DAJAT ResNet18 CIFAR10 ](https://drive.google.com/uc?id=1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f)
    * [TRADES-AWP WideResNet-34-10 CIFAR10](https://drive.google.com/uc?id=1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T)
    * [FAT ResNet50 ImageNet-1k](https://drive.google.com/uc?id=1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4)
    * [NuAT ResNet 18](https://drive.google.com/uc?id1-1DxecXz5U_xZ54DVdE-GVm71Tiox-Ri)

* The downloaded models are stored in the `checkpoints/`
* Other pretrained weights can be used from the [RobustBench model zoo](https://github.com/RobustBench/robustbench/tree/master/robustbench/model_zoo). The corresponding model code needs to be added to the `models/` directory (and modified similar to given example models).

## Neuron Importance Ranking Methods


* We first obtained the CLIP-Dissect and Leave-one-Out Neuron Importance Rankings from the following script files.
    * scripts/get_cdir_rankings.sh
    * scripts/get_loir_rankings.sh


## Few modifications:

* For the purpose of this project we made a few changes to few files:
    * In the get_loir_rankings.py file we replaced
            ' model_test.load_state_dict(torch.load('checkpoints/'+args.load_model)) '
            

            with 

            '# model_test.load_state_dict(torch.load('checkpoints/'+args.load_model))
            state_dict = torch.load('checkpoints/' + args.load_model)

            # If trained with DataParallel, keys will have "module." prefix — remove it
            if any(k.startswith("module.") for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace("module.", "")] = v
                state_dict = new_state_dict
            
            model_test.load_state_dict(state_dict)

            '

    * In the eval.py file we replaced 
            ' if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                n_examples = args.n_ex
                transform = transforms.Compose([transforms.ToTensor()])'

            with

            ' if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                n_examples = args.n_ex
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
                ])  '



## Analysis Experiment

* The analysis experiment (Fig. 2 in the paper) uses the LO-IR neuron importance rankings, so please run it first using `bash scripts/get_loir_rankings.sh`.
* After this, we can run the analysis experiment (by default for [DAJAT](https://arxiv.org/abs/2210.15318) ResNet18 pretrained model):


    ```
    bash scripts/analysis.sh
    ```

## AutoAttack Evaluation


* We ran the Standard Autoattack evaluation for the base model, CD-IR defended model and LO-IR defended model using the scripts/eval.sh 

* Adaptive attack evaluation will be released soon.

## Sources
* CLIP-Dissect: https://github.com/Trustworthy-ML-Lab/CLIP-dissect
* RobustBench: https://robustbench.github.io/

## Cite this work
A. Kulkarni and T.-W. Weng, Interpretability-Guided Test-Time Adversarial Defense, ECCV 2024.

```
@inproceedings{kulkarni2024igdefense,
    title={Interpretability-Guided Test-Time Adversarial Defense},
    author={Kulkarni, Akshay and Weng, Tsui-Wei},
    booktitle={European Conference on Computer Vision},
    year={2024}
}
```
