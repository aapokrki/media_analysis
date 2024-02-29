This folder will contain the original dataset, training data, testing data, validation data and the masks.

## Guide

**POTSDAM**

1. Download the potsdam dataset **"2_Ortho_RGB.zip"** **[HERE](https://drive.google.com/drive/folders/1w3EJuyUGet6_qmLwGAWZ9vw5ogeG0zLz)**
2. Extract the zip folder to ``./data/2_Ortho_RGB``

**MASKS**

1. Download the mask dataset **"test_mask.zip"** **[HERE](https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip?e=1&dl=0)**
2. Extract the zip folder to ``./data/test_mask``


If done correctly, the potsdam VHR dataset images should be in folder `````./data/2_Ortho_RGB````` 
AND the mask dataset images in folder ``./data/test_mask/mask/testing_mask_dataset``

**You can rename the ``./data/2_Ortho_RGB`` and ``./data/test_mask`` to anything you like, as long as you initialize the ImageProcessor class correctly**

Run the main function in ``image_processor.py``. 

The program will generate the training, validation and testing data from the 
dataset to the folders ``./data/train``,``./data/validation`` and ``./data/test`` with the ratios that are specified in the ImageProcessor class

**Default ratios:** 
- Train = 70%
- Validation = 15%
- Test = 15%

To test the ImageProcessor and see how it works, run and study ```test_image_processing.py```


   