# Robot-ASR using Wav2vec2

![alt text](https://github.com/brucew5978-git/Robot-ASR/blob/main/images/Storm_Iteration_2_-_Assembly.png)

## What is Robot-ASR?

Robot-ASR or Robot Automatic Speech Recognition (ASR) uses pretrained Wav2Vec2 - a Deep Learning Audio model introduced by Facebook AI Research, which is further finetuned on Timit - an acoustic speech dataset, in order to increase the ASR's effectiveness in speech recognition 

## What is it used for?

![alt text](https://github.com/brucew5978-git/wav2vec2-ASR/blob/main/images/audio-to-command.jpeg)

The Robot-ASR model is intended to be deployed on Trinity Robotics' Storm robot in order to give operation commands - like "go forward", using audio input. The predicted command can then be passed into the robot's logic node which can then perform appropriate actions to respond to the user's commands - such as setting motor speeds to move robot forward. 

However, this model is not limited to operations on Storm and can be deployed on any device that supports the necessary requirements. The model can also be quantized to reduce the file size and better suit operations on edge or mobile devices with limited storage space.

## How to use it?

1.  First run the setup.py file to download the ASR model
2.  Use audio-recording.py to record a audio file for sampling
3.  To feed the file into the machine learning model for evaluation - a process known as inference, run inference.py

Quantization (Optional)

Neural Networks on a high level, are networks with a large number of layers. Each layer contain a certain number of nodes which often uses floating point numbers to retain information. However, neural nets can be quite large and take up significant space to store which is a problem for edge devices - like single board computers, which are commonly used in robotics. 

Thus, the method of quantization can be utilized, which essentially converts a model's weights from floating point - 32 bit, to integers - 8 bit, effectively reducing the size of the model. Note that as weights lose resolution, performance of the quantized models may be somewhat decreased. 

4. Use model-quantization.py to decrease

## Performance

When evaluating the performance of this ASR model, the Robot ASR base and quantized variations were tested against the non-finetuned Wav2Vec2 base model. All three models were evaluated using a Timit test set of 1680 audio samples with Word Error Rate (WER) being used evaluate each model's accuracy. 

Models | WER | Avg Memory Usage, Gb | Inference Time (per file), s | 
--- | --- | --- | --- | 
Robot-ASR | 0.277 | 7.48 | 0.62 | 
Robot-ASR Quantized| 0.275 | 7.31 | 1.61 | 
Wav2Vec2-base| 0.360 | 8.42 | 0.63 | 

The evaluation results show that although both variations of Robot-ASR have higher speech inference accurracy compared to the base Wav2Vec2 model, both models also have slower inference speeds. Furthermore, the quantized model seems to have ... which is extremely undesirable for realtime applications like deploying on Trinity Storm. 

The large discrepancy in inference time between the quantized and unqauntized Robot-ASR models is likely due to the memory overhead caused by on the fly conversions utilized in dynamic quantization. Post-training Dynamic Quantization was used in this project for the following reasons: 
  * Post-training quantization was chosen as quantization aware training can be tricky and time consuming to figure out
  * Dynamic quantization was also chosen over Static due to its relatively easier implementation

Thus, for future model quantization, we will attempt to implement post-training static quantization to help decrease the model's inference times, as well as improve memory usage efficiency.

## Resources and References
https://www.trinityrobotics.ca

https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb#scrollTo=Adm0LngNNxq7

https://arxiv.org/pdf/2006.11477.pdf

https://huggingface.co/docs/optimum/concept_guides/quantization
