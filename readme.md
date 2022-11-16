## Auto Visualization of Auto Image Selection (AVAS)

### 1. Introduction
Visualization Codes of recall model for paper《What Image do You Need? A Two-stage Framework for Image Selection in E-commerce》. 
This is an extra project, and the main paper codes project is at `https://github.com/youyuge34/AutoImageSelection`.

This project includes a small image datasets(1k POI images)，pretrained model with weights.

### 2. MMKG content-based recall model

#### 2.1 Dataset and weights
`src/pretrain_model_test_dataset.csv` includes 1k images of 100 hot POI. 
You can replace it into the 5k dataset csv in the paper from our main project, we reduce the image file numbers here for convenience of visualization.
(5k image shown in a single html will case OOM.)
You should download the images into `src/images_test` using the url by yourself.


Secondly, download the weights file from `https://github.com/youyuge34/AVAS/releases/tag/v0.1`.
and then copy it to the dir weights. This weight file is totally the same as `align_model.pth` in our main project.

Now the dir is:
```
--AVAS
    --weights
        --commodity_poi_mml_20211221_with_MKG_withMKG_iteration_200000_0.4215_0.407_0.018.pth
```

#### 2.2 Quick Start
```commandline
pip install -r requirements.txt
cd demo
python3 align_model_test_with_mkg.py
```

- `demo/align_model_test_with_mkg.py`     # the demo test file
- `model/image/align_model_with_MKG.py`   # the model structure file
- `src/images_test`   # test images dir (1k number here)
- `weights/`    # weights of pretrained model encoders and Bert config

The demo test file `align_model_test_with_mkg.py` will load the encoders weight and calculate the relevant score between the
text and the images. Then it will save the results to file `align_普陀山图片_with_MKG.csv`. You can learn more from the python file
and feel free to edit it. 

Then you can edit and run the visualization script which takes `align_普陀山图片_with_MKG.csv` as input:
```commandline
python3 visualzation_csv.py
```
Now you can see the generated html file which can be opened using Chrome:

- Visualizaiton of text-image retrieval. The higher score denotes the image is more relevant to the POI.
![high score results](demo/result1.jpg)
  
![low score results](demo/result_low_score.jpg)

We can find that the low score images are irrelevant to the POI '普陀山'。

