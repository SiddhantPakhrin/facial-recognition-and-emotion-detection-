# facial-recognition-and-emotion-detection-
Real time face recognition and Emotion detection using deep learning

for description read https://github.com/atulapra/Emotion-detection

i have motified the above mentioned project from atula pra and added database as well as facial recognition feature. It asks the user to enter id and their name as well as age and gender.

After you have downloaded all the necessary files and installed necessary dependencies ..

Then run:
```bash
cd src
py datasetcreator.py
```
This creates your dataset and takes 100 inputs of your face and saves it in the dataSet folder.

Then run:
```bash
py trainer.py
```
This trains your dataset.
After that you can run 

```bash
cd src
py emotions.py --mode display
```

For the database install Sqlite studio...load the database file in the Sqlite studio where you can see your changed/updated data..
