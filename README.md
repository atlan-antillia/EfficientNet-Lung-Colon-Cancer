<h2>EfficientNet-Lung-Colon-Cancer (Updated: 2022/09/10)</h2>
<a href="#1">1 EfficientNetV2 Lung Colon Cancer Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Lung Colon Cancer dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Lung Colon Cancer Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Lung Colon Cancer Classification</a>
</h2>

 This is an experimental EfficientNetV2 Lung Colon Cancer Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>
 The original lung_colon_image_set has been taken from the following web site:<br>

<a href="https://www.kaggle.com/code/tenebris97/lung-colon-all-5-classes-efficientnetb7-98/data">
Lung & Colon ALL 5 Classes | EfficientNetB7 | 98%%
</a>
</a>

<b>About This Data:</b><br>
<pre>
This dataset contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format.
The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the Augmentor package.
There are five classes in the dataset, each with 5,000 images, being:

Lung benign tissue
Lung adenocarcinoma
Lung squamous cell carcinoma
Colon adenocarcinoma
Colon benign tissue
How to Cite this Dataset
If you use in your research, please credit the author of the dataset:

Original Article
Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019

Relevant Links
https://arxiv.org/abs/1912.12142v1
https://github.com/tampapath/lung_colon_image_set
Dataset BibTeX
@article{,
title= {LC25000 Lung and colon histopathological image dataset},
keywords= {cancer,histopathology},
author= {Andrew A. Borkowski, Marilyn M. Bui, L. Brannon Thomas, Catherine P. Wilson, Lauren A. DeLand, Stephen M. Mastorides},
url= {https://github.com/tampapath/lung_colon_image_set}
}

</pre>

<br>
<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<br>

<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/EfficientNet-Lung-Colon-Cancer.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Lung-Colon-Cancer
        ├─eval
        ├─evaluation
        ├─inference        
        └─test
</pre>
<h3>
<a id="1.2">1.2 Prepare Lung Colon Cancer dataset</a>
</h3>

Please download the dataset <b>Lung and Colon Image dataset</b> from the following web site:
<br>
<a href="https://www.kaggle.com/code/tenebris97/lung-colon-all-5-classes-efficientnetb7-98/data">
lung_colon_image_set</a>
</a>
<br>
The <b>lung_colon_image_set</b> has two sub directories, <b>colon_image_sets</b> and <b>lung_image_sets</b>:<br>
<pre>
lung_colon_image_set
  ├─colon_image_sets
  │  ├─colon_aca
  │  └─colon_n
  └─lung_image_sets
      ├─lung_aca
      ├─lung_n
      └─lung_scc
</pre>
To simplify a dataset, we have created <b>Lung_Colon_Images-master</b> dataset from the <b>lung_colon_image_set</b> above:<br>
<pre>
Lung_Colon_Images-master
  ├─colon_aca
  ├─colon_n
  ├─lung_aca
  ├─lung_n
  └─lung_scc
</pre>
Futhermore, we have created <b>Lung_Colon_Images</b> dataset by splitting the master dataset to
 train and test sets by <a href="./projects/Lung-Color-Cancer/split_master.py">split_master.py</a>.
<br>
<pre>
Lung_Colon_Images
  ├─test
  │  ├─colon_aca
  │  ├─colon_n
  │  ├─lung_aca
  │  ├─lung_n
  │  └─lung_scc
  └─train
      ├─colon_aca
      ├─colon_n
      ├─lung_aca
      ├─lung_n
      └─lung_scc
</pre>
<br>
<br>
The number of images in classes of train and test sets:<br>
<img src="./projects/Lung-Colon-Cancer/_Lung_Colon_Images_.png" width="740" height="auto"> 
<br>
<br>
Sample images of Lung_Colon_Images/train/colon_aca:<br>
<img src="./asset/Lung_Colon_Images_train_colon_aca.png" width="840" height="auto">
<br>
<br>

Sample images of Lung_Colon_Images/train/colon_n:<br>
<img src="./asset/Lung_Colon_Images_train_colon_n.png" width="840" height="auto">
<br>
<br>

Sample images of Lung_Colon_Images/train/lung_aca:<br>
<img src="./asset/Lung_Colon_Images_train_lung_aca.png" width="840" height="auto">
<br>
<br>

Sample images of Lung_Colon_Images/train/lung_n:<br>
<img src="./asset/Lung_Colon_Images_train_lung_n.png" width="840" height="auto">
<br>
<br>

Sample images of Lung_Colon_Images/train/lung_scc:<br>
<img src="./asset/Lung_Colon_Images_train_lung_scc.png" width="840" height="auto">
<br>
<br>


<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Lung Colon Cancer Classification</a>
</h2>
We have defined the following python classes to implement our Lung Colon Cancer Classification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train Lung Colon Cancer FineTuning Model.
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─Lung-Colon-Cancer
  ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Lung Colon efficientnetv2 model by using
<b>Lung_Colon_Images/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=384 ^
  --eval_image_size=480 ^
  --data_dir=./Lung_Colon_Images/train ^
  --data_augmentation=True ^
  --valid_data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --num_epochs=50 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 60
horizontal_flip    = True
vertical_flip      = True 
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.2, 2.0]
data_format        = "channels_last"

[validation]8
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 60
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.3, 2.0]
data_format        = "channels_last"
</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Lung-Colon-Cancer/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Lung-Colon-Cancer/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Lung-Colon-Cancer_train_console_output_at_epoch_50_0910.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/Lung-Colon-Cancer/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Lung-Colon-Cancer/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the breast cancer in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --image_path=./test/*.jpeg ^
  --eval_image_size=480 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
colon_aca
colon_n
lung_aca
lung_n
lung_scc
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Lung-Colon-Cancer/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Lung-Colon-Cancer/Lung_Colon_Images/test">Lung_Colon_Images/test</a>.
<br>
<img src="./asset/Lung_Colon_Images_test.png" width="840" height="auto"><br>


<br>
<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Lung-Colon-Cancer/inference/inference.csv">inference result file</a>.
<br>
<br>
Inference console output:<br>
<img src="./asset/Lung-Colon-Cancer_infer_console_output_at_epoch_50_0910.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Lung-Colon-Cancer_inference_at_epoch_50_0910.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Lung-Colon-Cancer/Lung_Colon_Images/test">
Lung_Colon_Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./Lung_Colon_Images/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --eval_image_size=480 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Lung-Colon-Cancer/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Lung-Colon-Cancer/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Lung-Colon-Cancer_evaluate_console_output_at_epoch_50_0910.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Lung-Colon-Cancer_classification_report_at_epoch_50_0910.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Lung-Colon-Cancer/evaluation/confusion_matrix.png" width="740" height="auto"><br>


<br>
<h3>
References
</h3>
<b>
A Machine Learning Approach to Diagnosing Lung and Colon Cancer Using a Deep Learning-Based Classification Framework
</b><br>
<pre>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7865416/  
</pre>
<br>
<b>Lung & Colon ALL 5 Classes | EfficientNetB7 | 98%</b> <br>
<pre>
https://www.kaggle.com/code/tenebris97/lung-colon-all-5-classes-efficientnetb7-98/data
</pre>
<br>
