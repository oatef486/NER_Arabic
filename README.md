# Arabic NER Model

**Engineering Teamwork Project**

# Authors:

1- Omar Saleh -Matriculation number: 3120009

3- Ibrahim Khalil -Matriculation number: 3119779

# arabic_ner.py:

- This the first pre-trained model we adopted at the beginning of this project.
- We used a powerful pre-trained model developed by Stanford NLP Group, it is designed for multiple NLP capabilities including NER for the arabic language.

# label_identifier.py:

- This file is used to identify the encoding of the dataset we obtained online from Masader website.
- It also identifies the unique labels present in the dataset and their numbers , showing how many times they occured.
- It also shows the tags the authors used for organising the named entities in the pre-annotated dataset, which is important in the tokenisation process.

# tokenization.py:

- This file is used to read the dataset file.
- It also is used to replace underscores with spaces.
- It tokenizes the data.
- The Camel Tools pre-trained NER model is used to predict named entities in the text.

# splitting_dataset.py

- The dataset when downloaded in a zipfile format, so this code is used to unzip the file.
- It uses the data provided from label_identifier.py file, to check which encoding was used by the original authors of the dataset
- We wrote this code, so that it splits the dataset into two parts. one part is used for the training of the model (which was 90% of the dataset) and the other part was used for the evaluation of the model (which was the rest 10% of the dataset).

# google_colab_train.py

- We ran this code on google colab. As all the before files were ran on visual studio code. So it consists of three codes put in one file with big spaces between the three of them.
- The first part is used to unzip the zipped dataset (this is repeated again, because the code was ran on google colab and wasn't really linked to the splitting_dataset file), it also identifies and counts all unique tags in the dataset, and finally it saves these unique tags to a file.
- In the second chunk of code the unique tags that we saved in the step above are loaded and a map for the labelling in created and then is saved in a JSON file
- In the third chunk of code is used to train the model by loading the JSON file to identify the unique mapping identified and then the training process commences. The code also evaluates the Precision, recall anf F1 score of the model every 200 steps.
- A problem we faced was that the dataset was huge for our laptops GPUs to handle, that was the main reason we used google colab services as we could run the training process on google servers. we used A100 Nivida GPUs, which decreased the process's time significantlly.
- Another problem we faced was that the dataset was still so huge for the runtime google colab allocated even for the premium users, thats why we added checkpoints, so in the case that the training process is interrupted for any particular reason,it canresume from where it stopped. which saved a lot of time.

# run_model.py:

- This fileis used to run the model after the training process.
- for factual texts pasted in the text section, named entities should be identified.
- However, we faced issues with actually identifying named entities as the precision of the model isn't the best, but it still gets some named entities correctly.

# Additional notes:

- The trained model is about 88 Gb big and is saved on Omar's google drive, for access permissions, please E-mail Omar about it.
- Some of the code is repeated as some of it is written to be used on visual studio code and the rest is written to be used on google colab. This is because when we started programming this project we weren't aware of how GPU and proccessor intensive the training process is. So we had to use the more advanced and powerful google GPUs, which is why some of the code won't work on visual studio code, but works on google colab and vice versa.
- For any questions about which scripts work on which platform, dont hesitate to contact us.
