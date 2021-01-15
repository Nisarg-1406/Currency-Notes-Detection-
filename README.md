# Indian-Currency-Notes-Detection
Aim is to Detect the different currency notes of India

## Table of Contents - 
* [About Project](#about-project)
* [Detailed Explanation about Project](#detailed-explanation-about-project)
* [Output Images](#output-images)
* [About Me](#about-me)

## About Project
* The Project aims for detecting the different currency Notes of India and giving the correct output. The purpose of the project is that, As there are physically disable people in our country, this project would help to identify different currency notes. For Eg : For the blind person, When they are doing money transaction, as they are not able to see, this project would help them to do the transactions. 
* How this Project Useful - I have done research and found that, Person without seeing can **hold the currency notes in 8 different forms**. I have shown the image below considering 8 different forms. The computer vision Algorithm i.e single shot mulitbox detector detects the currency notes in all 8 directions forms and hence making it Easier for the Blind person for doing the Transactions.  

## Detailed Explanation about Project
**I HAVE GIVEN DETAILED EXPLANATION OF EACH AND EVERY LINE OF MY CODE**

* First we would link the Google drive with the Google Colab as we are running the code on Google colab for faster processing due to usage of GPU. Also the version used here for tensorflow is 1.15.2
   ```
   from google.colab import drive
   drive.mount('/content/drive')
   %cd '/content/drive/My Drive/Monuments Of India/'
   ```
   
* Then will define the number of training steps - `1000` and number of the evaluation steps - `50`. Thr number of evaluation step is to check the model performance on non train data. Now we would define the model configuration, so I would be using `SSD Mobile Net V2 configuration` - SSD is single shot multibox detector. SSD is thr type of the object detection technique which has reached new records in terms of performance and precision for object detection tasks, scoring over 74% mAP (mean Average Precision) at 59 frames per second on standard datasets such as PascalVOC and COCO. Single Shot: this means that the tasks of object localization and classification are done in a single forward pass of the network. MultiBox: this is the name of a technique for bounding box regression. Detector: The network is an object detector that also classifies those detected objects. 
  ```
  'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
  }
  ```
  
* Then I have taken `80:20` train vs test images. With this images, I have generated the XML files and then converted train folder annotation xml files to a single csv file. Same is done with test folder annotation xml files to a single csv file.
    ```
    !python xml_to_csv.py -i data/Images/train -o data/annotations/train_labels.csv -l data/annotations  ---> For Train Images
  	!python xml_to_csv.py -i data/Images/test -o data/annotations/test_labels.csv --------------------------> For Text Images
    ```

* Next is to generate the TFRecords for both train and text images csv files. The TFRecord format is a simple format for storing a sequence of binary records. For future using purpose we need to store up the train record name as train_record_name & test record name as the test_record_name.
    ```
    !python generate_tfrecord.py --csv_input=data/annotations/train_labels.csv --output_path=data/annotations/train.record --img_path=data/Images/train --label_map data/annotations/label_map.pbtxt
    !python generate_tfrecord.py --csv_input=data/annotations/test_labels.csv --output_path=data/annotations/test.record --img_path=data/Images/test --label_map data/annotations/label_map.pbtxt
    ```

* Now to download the Mobilenet SSD v2 Model and then we would be setting TensorFlow pretrained model checkpoint for better proecessing of the model
  ```
  fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
  fine_tune_checkpoint    
  ```
 
* Then we would be Configuring a Training Pipeline, so first we would be joining the pipeline file to already existing folder of `/content/models/research/object_detection/samples/configs/`, giving the name - `pipeline_fname` and then with use of `assert` we would be checking if `pipeline_fname` exist or not. The `assert` statement simply takes input a boolean condition, which when returns true doesn’t return anything, but if it is computed to be false, then it raises an AssertionError along with the optional message provided. So here the error message is printed using `.format()` in python. `str.format()` is one of the string formatting methods in Python3, which allows multiple substitutions and value formatting. This method lets us concatenate elements within a string through positional formatting. Its syntax is `Syntax : { } .format(value)` -> (value) : Can be an integer, floating point numeric constant, string, characters or even variables.
    ```
    import os
  	pipeline_fname = os.path.join('/content/models/research/object_detection/samples/configs/', pipeline_file)
    assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)
    ```
