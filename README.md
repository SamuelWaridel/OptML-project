# CS-439 : Optimization for Machine Learining - project
## Exploring generalization capacity of SGD, AdaGrad and Adam

Contributors : Tallulah Rytz, Timothée Chaadi Coester, Samuel Waridel


### Project Description

The goal of this project is to explore the generalization capacity of three widely used optimization algorithms (SGD, Ada-Grad, and Adam) in the context of image classification on the CIFAR-10 dataset. To do so, we compared three different architectures for each of these algorithms. Here we have provide the code that was used for this analysis.

The models were trained on the CIFAR-10 dataset for image classification, and a grid search algorithm was used to tune the hyperparameters. The trained models were then evaluated using the corrupted images form the CIFAR-10-C dataset, as well as being subjected to black box attacks.

The final results help indicate the differences between these three algorithms, as well as point to the need of further research to fully understand their strenghts and weaknesses.


### Code Description

In the main directory of the project is a file called run.py. This file contains the script that was used to obtain the final results for this project.

The model training is ommited as it would take much too long to be practical. The best models are simply loaded from the provided .pth files in the Best_model/Models subfolder. A synthesis of the training is provided under Best_models/Training results.

Once loaded, models are then evaluated on the CIFAR-10-C dataset. This dataset consists of corrupted versions of the original CIFAR 10 images with different severities, and they are loaded automatically from the original Zenodo. This allows to test for the models resistance to such perturbations, and the results are saved under Best_Models/Corruption evaluation.

After that, the models are subjected to black box attacks through. For simplicity and because of the time constraints, only one kind of attack was considered (Boundary Attacks). Images are subsampled from the original dataset, and the model's accuracy is recorded before and after perturbation, as well as a mean perturbation size for each one. The results are placed in the BlackBoxAttack.csv file under Best_models.

Finally, the code loads the computed resutls into a dataframe for plotting and visualization.

Even when removing training, the black box attacks take quite some time, therefore the script provides an option to skip the evaluation steps and simply load the pre-computed results.

Most of the custom functions that were made, as well as the model definitions, are under Functions/implementations.py. This was done to keep things a bit cleaner. the plotting functions were placed under Functions/visualization.py. The pngs are saved under Best_Models/Figures.

We also provide Jupyter notebooks (in the Notebooks folder) that were used to do most of the training and evaluation. The final run.py file can be seen as a regrouping of all of these notebooks together.

### How to use the Code

To use the code, the first thing is to make sure to have the proper dependencies loaded. We have provided a requirement.txt file to make creating the right environmetn much simpler. A brief description of the necessary packages is also provided below.

The main block of code is in the run.py file. When running this script, the user will be first asked if they want to perform the corruption evaluation and black box attack, or skip this step and simply plot the pre-computed results. After answering this question, the program proceeds with the desired process while providing feedback to the user.

If one wishes to have a more hands on view view, we have provided Jupyter notebooks to show in more detail how the code is constructed.

### Dependencies and Imports

This code was run locally with Python 3.13.2. Most of the training was done using Google Colab's web services, in order to have access to GPUs and faster computation time.

The full list of necessary requirements for running this code can be found in the requirements.txt file.

The most important packages are listed as follows: 

- numpy : 2.2.5
- pandas : 2.2.3
- matplotlib : 3.10.3
- scikit learn : 1.6.1
- pytorch : 2.6.0
- torchvision : 0.21.0
- foolbox : 3.3.1
- jupyter : 1.1.1
- requests : 2.32.3

Less critical packages include: 
- seaborn : 0.13.2
- scipy : 1.15.2
- IPython : 8.30.0
- ipywidgets : 8.1.5
- tqdm : 4.67.1