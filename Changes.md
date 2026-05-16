# Main changes to the orginial repository for improvement II by Nikan Mahdavi Tabatabaei

The files with the prefix "Mix_Dist", "Changed" and "Plot" in the experiments folder are made by me.

The python files of type distorted_training_b_mobilenet_caltech with the prefix "Changed" are the original codes with the bugs fixed, while the ones with the prefix of "Mix_Dist" are the most important ones which have my improvement included in them. The ones who also have a later prefix of "GPU" are used for training the model on the GPU cluster. 

Furthermore the Mix_Dist_generate_distortion_classifier_dataset, Mix_Dist_distortionNet and Mix_Dist_training_distortion_classifier, also do exactly what the original code did (in regards to the distortion classifier) with the addition and integration of our improvement. As a simple example the Mix_Dist_distortionNet now includes 4 output classes rather than 3. Otherwise the codes are similar, we have just fixed quite some bugs from the original and integrated our improvements. 

The "plot"-prefixed files in the experiments are also made by me, which repoduce paper results, while also adding and integrating our improvement into it for our specific plots in our paper/project. 