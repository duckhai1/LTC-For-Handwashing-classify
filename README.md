# LTC-For-Hashwashing-classify
Using handwashing dataset for assess LTC

Dataset source: https://www.kaggle.com/realtimear/hand-wash-dataset

LTC reference: https://github.com/raminmh/liquid_time_constant_networks/tree/master

# How to run
- Step 1: 

> `python read_data.py`

This command will invoke pre-processing data and store the clean data in "clean_data" folder. Each data is a matrix of size **(number_of_frame, number_of_feature)**. Where number_of_frame is determine by *PROCESS_VIDEO_LENGTH* (default 30) and *FRAME_STEP* (default 3), which means we take frame number 0,3,6,...,30 

- Step 2:

> `python train.py --model ltc_ex --epochs 200 > ../result/ltc_ex_200e.log`

This command will start the training process and store result in log file at "result" folder. The accuracy is determine by assess each input video represent each step is label correctly. Here are some option model to train:

    - lstm
    - ltc
    - ltc_ex
    - ltc_rk
    - node
    - ctgru
    - ctrnn
