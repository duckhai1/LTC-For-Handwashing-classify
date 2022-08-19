# LTC-For-Hashwashing-classify
Using handwashing dataset for assess LTC comparing with other variation of RNN

Dataset source: https://www.kaggle.com/realtimear/hand-wash-dataset

LTC reference: https://github.com/raminmh/liquid_time_constant_networks/tree/master

# How to run

### Train model

- Go to src folder then:
    1. Train the inner layer (i.e layer1).

        > `python main.py -l layer1`

        Layer_1 input data shape is **(number_of_frame, batch_size, number_of_feature)**. 
        
        Where **number_of_frame** is determine by *PROCESS_VIDEO_LENGTH* and *FRAME_STEP*, which means we take frame number 0,3,6,...,30. **batch_size** is the number of data sample pass in 1 time. **number_of_feature** is size of vector to represent each frame
    2. Train the outer layer (i.e layer2). 

        > `python main.py -l layer2`

        Layer_2 input data shape is **(number_of_sample, 12)**.

        Where the second axis is contain likelihood percentage of 6 steps and then score of 6 step in order.
        
- You can specify to enable train/test mode by --train / --test flag. Default both of them is true

- You can also specify the custom path containing collection of train dataset or test dataset only through --trainData / --testData input 

- The property list is in **/src/property.py**. 

### Evaluate whole project with single data  
Turning on the --eval flag and feed video path to --path

> `python main.py --eval true --path <video_path>`

### Read result
* The result of each network tree can be found in location: *results/<**layer1_model_type**>/<**layer1_model_size**>\_<**layer1_model_type**>\_<**tree_id**>/csv_file*

* The result of layer 1, full random forest is stored in: *results/<**layer1_model_type**>/forest_size<**layer1_model_size**>.csv

* The result of layer 2 is stored in: *results/<**layer1_model_type**>/<**layer2_model_type**>\_<**layer2_epoch**>_result.csv
