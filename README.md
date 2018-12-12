# Tim Sternberg | CS398 F18 Neural Network Independent Study



## INSTALL INSTRUCTIONS

Installing Tensorflow: https://www.tensorflow.org/install/

Installing Recurrent Network: https://www.tensorflow.org/tutorials/sequences/recurrent

## RUN INSTRUCTIONS

python kdd_word_lm.py --data_path=F:/tensorflow/kdd_rnn --model=small --percent_of_data=10

To run tensorboard ensure the save path is specified correctly in kdd_word_lm.py then run this command: python -m tensorboard.main --logdir=F:/tensorflow/kdd_rnn/save

And then go to this address in your internet browser: http://localhost:6006