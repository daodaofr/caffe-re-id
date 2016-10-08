1. Download data from https://drive.google.com/file/d/0ByS8YXR7ycXHMGtJSkRLQUVlcmM/view or http://pan.baidu.com/s/1dF3HyGp, which contains the lbp&color feature of PRID dataset and the trained model.
2. Run ex_fea_sequence.py to extract the learned features, you can directly use the trained model: re-id.caffemodel. You need to configure the color_path to your local path containing lbp&color feature.
3. If you want to train your own model, create folder "log" and "model" before training, which is used to save log files and models. Excute run_lstm.sh for traning the lstm network, you need to configure fea_path in sequence_input_layer.py wrt your local path containing lbp&color feature.

