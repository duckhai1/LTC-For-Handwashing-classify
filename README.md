# LTC-For-Hashwashing-classify
Using handwashing dataset for assess LTC comparing with other variation of RNN

Dataset source: https://www.kaggle.com/realtimear/hand-wash-dataset

LTC reference: https://github.com/raminmh/liquid_time_constant_networks/tree/master

# How to run

### Train model

- Go to src folder then:
    1. Preprocessing the data (update these properties `REGENERATE_LAYER1_DATA` and `REGENERATE_LAYER2_DATA` to **True** )
        
        > `python main.py -l preprocess`

        This will create a list of data of both layer1 and layer2 for train/valid/test dataset, the list is written in file `.lst`. Then from this list it will generate the data object in folder `data/layer_<1/2>_data` (and also processing the data for layer2 case).

        For layer1, the data to pass into model is located in `LAYER_1_TRAIN_PATH` 

        For layer2, the data to pass into model is located in `LAYER_2_TRAIN_PROCESSED_PATH`

        We can disable the shuffle feature to use the pre-existed `.lst` object by adding argument `--divideLayer1 false --divideLayer2 false` and set both properties `REGENERATE_LAYER1_DATA` and `REGENERATE_LAYER2_DATA` to **False**. The pre-existed data list will be stored in `LAYER_1_TRAIN_FILE` and `LAYER_2_TRAIN_FILE` (refer `properties.py` file for more configuration)

    2. Train the inner layer (i.e layer1).

        > `python main.py -l layer1`

        This method will use for start training and testing the layer 1 model.

        Layer_1 input data shape is **(number_of_frame, batch_size, number_of_feature)**. 
        
        Where **number_of_frame** is determine by *PROCESS_VIDEO_LENGTH* and *FRAME_STEP*, which means we take frame number 0,3,6,...,30. **batch_size** is the number of data sample pass in 1 time. **number_of_feature** is size of vector to represent each frame

        We can specify to disable train/test mode by adding flag `--train false --test false`. Default both of them is true

    3. Train the outer layer (i.e layer2). 

        This method will use for start training and testing the layer 1 model.

        > `python main.py -l layer2`

        Layer_2 input data shape is **(number_of_sample, 12)**.

        Where the second axis is contain likelihood percentage of 6 steps and then score of 6 step in order.
        
        We can specify to disable train/test mode by adding flag `--train false --test false`. Default both of them is true


- The property list is in **/src/property.py**. 

### Evaluate whole project with single data  
Turning on the --eval flag and feed video path to --path

> `python main.py --eval true --path <video_path>`

### Read result
* The result of each network tree can be found in location: *results/logs/<**layer1_model_type**>/<**layer1_model_size**>\_<**layer1_model_type**>\_<**tree_id**>/csv_file*

* The result of layer 1, full random forest is stored in: *results/logs/<**layer1_model_type**>/forest_size<**layer1_model_size**>.csv

* The result of layer 2 is stored in: *results/logs/<**layer1_model_type**>/<**layer2_model_type**>\_<**layer2_epoch**>_result.csv

* The model parameter can be found ind: *results/tf_sessions/*
    * The **model-** folder is the trees parameter saves
    * The **.pkl** file is the layer2 parameter saves 
