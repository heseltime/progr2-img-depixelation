# Progr2 ML project: Image Depixelation Challenge

![image](https://github.com/heseltime/progr2-img-depixelation/assets/66922223/b77b053e-d32e-4b5f-bc86-64042590cbec)

This project is an example ML project developed for the Johannes Kepler University Python Programming 2 course (AI Bachelors/Masters-requirement). The model is used to particpate in the annual machine learning challenge posted by the Institute of Machine Learning.

### Example usage

Training
```
python main.py .\working_config.json
```

Inference
```
python predict.py
```

### Structure
```
progr2-img-depixelation
|- architectures.py
|- assignments folder
|   contains all the assignment functions and datastructures developed over the semester (without assignment sheets)
|- doc folder
|   additional images and other materials for documentation
|- example-project-past-yr folder
|   project template from last year (I took the course then but dropped it about halfway through the semester)
|- main.py
|    Main file implementing training loop and evaluation
|- predict.py
|   Produces the submission.pkl using ...
|- submission_serialization.py
|   Given in the course, deserializes ...
|- test_set.pkl
|- results folder
|   Saves the output from main.py, mainly scores as .txt and the .pt model file
|- utils.py
|    Utility functions and classes
|- working_config.json
|    last configuration used 
|    (see also alternative configurations: past_config_*.json where * is some feature of the training)
|+ requirements.txt
```

### Dependencies
See requirements.txt

### Results

#### Test submission 0: heseltest0 (hey, not last on the leaderboard!)

![image](https://github.com/heseltime/progr2-img-depixelation/assets/66922223/fab3d824-9a1b-4cac-8bb0-b7e93f2d3ab1)

Simple Network architecture, minimal training: note it is actually worse than the IdentityModel (uses input as prediction, i.e., it does
not perform any actual computation). There's some room for improvement: estimated points for the project is 0 (see Scoring Scheme).


### Scoring Scheme

The reference BasicCNNModel is a model implementing 5 layers with 32 kernels of size 3. If the submission model’s RMSE is equal to or higher than IdentityModel, the scoring for the project is 0 points. If the model’s RMSE is equal to BasicCNNModel, it is 200 points. Everything in between the two models is linearly interpolated, with bonus points for better than the reference model. 