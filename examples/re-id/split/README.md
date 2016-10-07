1. Download data from , which contains the lbp&color feature of PRID dataset and the trained model.
2. run ex_fea_sequence.py for extracting features, you can directly use the trained model: re-id.caffemodel
3. If you want to train your own models, create folder "log" and "model" before training, which is used to save log files and models. Excute run_lstm.sh for traning the lstm network, you may need to configure the path to the lbp_color feature.

