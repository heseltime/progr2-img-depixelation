# Progr2 ML project: Image Depixelation Challenge

![image](https://github.com/heseltime/progr2-img-depixelation/assets/66922223/b77b053e-d32e-4b5f-bc86-64042590cbec)

This project is an example ML project developed for the Johannes Kepler University Python Programming 2 course (AI Bachelors/Masters-requirement). The model is used to particpate in the annual machine learning challenge posted by the Institute of Machine Learning.

### Example usage

#### Training
```
python main.py .\working_config.json
```

#### Inference
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
|- past_main.py (past_*)
|   Initital tests, mainly implementing a simple linear Neural Network architecture, before doing the CNN.
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

#### heseltest0 (test submission 1 with simple Network architecture: hey, not last on the leaderboard!)

![image](https://github.com/heseltime/progr2-img-depixelation/assets/66922223/fab3d824-9a1b-4cac-8bb0-b7e93f2d3ab1)

Simple Network architecture, minimal training (1000 updates): note it is actually worse than the IdentityModel (uses input as prediction, i.e., it does not perform any actual computation). There's some room for improvement: estimated points for the project is 0 (see Scoring Scheme) - metric: 36.291  

#### heseltest0.1 (test submission 2: more training, same architecture as before)

![image](https://github.com/heseltime/progr2-img-depixelation/assets/66922223/688df73b-295b-41f8-9fdb-318f146571d9)

Really a toy architecture: the idea for the first proper attempt is to implement the CNN and use the configuration file settings for testing variants (aim for the three remaining attempts). Submission 2 with 100000 updates on a really simply architecture only brought marginal improvements on the leader board in terms of the metric, but a lower position because of newer, better submissions (still below identity however) - metric: 33.799

#### heselCNN1 - 2/3 (test submission 3 - 4/5: CNN architecture)

Just the metrics now, for a basic CNN implementation (see current main.py)

heselCNN1 metric: 21.891 (beats Identity (23.163): Still off from BasicCNN (17.892) however - 10000 update steps)
heselCNN2 metric: 25.768 (looks like classic overfitting, so more complex model worse thans simpler model) - 100000 update steps: took a work day to train on my laptop.
heselCNN3 metric: skipped for now, I want to keep this submission option open. (heselCNN2 final for now.)

So this is a model that solves this semster-long project in an ok, but not great, manner, using a basic Convolutional Neural Net.

### Next Steps/Critique

For this Project I will stop here: there is of course room for improvement in terms of complexity. The next step would have been to experiment with modern/advanced architectures. On the data side, one core idea would be to augment data to increase the training capability significantly.

### More on the Scoring Scheme

The reference BasicCNNModel is a model implementing 5 layers with 32 kernels of size 3. If the submission model’s RMSE is equal to the IdentityModel, the scoring for the project is 0 points. If the model’s RMSE is equal to BasicCNNModel, it is 200 points. Everything in between the two models is linearly interpolated, with bonus points for better than the reference model. 
