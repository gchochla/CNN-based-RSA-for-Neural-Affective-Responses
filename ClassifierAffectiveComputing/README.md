# Classifier

## Based on the dataset, train_model_v3.py and test_model.py code should be modified!!!

## Python version
    3.9.7

## Libraries
    Check the requirements.txt file.

    Example: pip install -r requirements.txt

## Train A Model:

    python train_model_v3.py -d <training data dir> -dv <validation data dir> -nt <num files of train data> -nv <num files of val data> [-m] <model name> [-e] <num epochs> [-b] <batch size> [-s] <sequence length> [-lr] <learning_rate> 

add flag -l to load a model from a checkpoint file

    Example: python train_model_v3.py -d ./data -dv ./data -nt 1 -nv 1 -m InceptionV3 -e 1 -b 1 -s 10 -lr 0.01 
    
Model Names:  CNN, AlexNet, AlexNetV2, Xception, ResNet50 or InceptionV3


## Model Testing

### Test CNN Models
Must have game running in a 800x600 window in the top left corner of the screen
    
    python test_model.py <absolute_path_to_saved_model>

### Test LSTM Models

    python test_model_lstm.py <seq_len> <model_name> <absolute_path_to_saved_LSTM_model>

Model names: ObjectDetection or ConvLSTM

## GPU or CPU
If you want to use GPU, please remove this below. The code is the first part of the main function.
```
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
```
