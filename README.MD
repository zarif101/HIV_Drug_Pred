# HIV Inhibition Prediction

HIV is an infectious disease that disrupts the lives of millions of people. Accordingly, there are
many chemical compounds which have been used to try to inhibit it's replication. In this repo, I use a dataset of compounds given in SMILES format,
each with a label of 0 or 1 (non-inhibiting or inhibiting), to create a deep learning model which can take in a compound and make the relevant binary inhibition
prediction!

The model is created and trained using Pytorch, and the dataset (curated by the Pande research group @ Stanford) can be found here: http://moleculenet.ai/

### Notes
- There is a class imbalance in the dataset, as there are many more compounds which cannot inhibit HIV replication (label 0) than those which can (label 1)
- To deal with the class imbalance, I'm currently using the Pytorch WeightedRandomSampler
- Util.py contains assorted helper functions
- 'notebooks' is a folder containing notebook(s) for visualization the dataset, and in the future, I'll add some which go through the model training process
- 'train_model.py' can be run to train the Pytorch model; test metrics will be printed out, and training history is saved in a .txt file
- Issue: Data preprocessing uses a library called DeepChem, which produces some useless console warnings, which is annoying and clogs up the terminal; I'm currently looking into workarounds or replacements
- I'll update this project and documentation as I have time
- Feel free to reach out to me via email at zarif.azher@gmail.com, or just open an issue right here on GitHub
