#Understanding the correlations within concentrated electrolytes using variational inference.

This repository contains code to generate the smoothed or standard radial distribution functions, 
used in my MSc Project. 
It also contains my implementation of the Variational Mixture of Gaussians, outlined by C. Bishop in C10,
"Pattern Recognition and Machine Learning". 

#Feature extraction
```descriptor_generator.py``` takes an input a series of csv files. These files are formatted such 
that the second, third and fourth columns correspond to the x, y and z coordinates of a 
particular particle and the fifth column corresponds to the label of that particle. 

An example of how to generate descriptors from csv files in this format:
```buildoutcfg
python descriptor_generator.py --directory='1.0-80/' --dataset=1080 --num_files=20 --smoothed=True
```

#Variational Mixture of Gaussians
An example of how to run the Variational Mixture of Gaussians code on prepared data:
```buildoutcfg
python inference.py 
```

#Requirements
- numpy (1.18.1)
- pandas (1.0.1)
- scikit-learn (0.22.1)
- scipy (1.4.1)


