AI Engineer (m/f/x) Assessment at payever

Documentation
Approach Taken:

Data Generation: Generated a synthetic dataset of products with attributes like name, description, price, and category using Python.

Data Preprocessing: Tokenized the descriptions and encoded the categories to prepare the data for training a machine learning model.

Model Development: Built a simple neural network model using TensorFlow to classify product categories based on text descriptions. The model consists of an embedding layer, a global average pooling layer, and dense layers.

Training and Evaluation: Trained the model with the generated dataset and evaluated using accuracy as the metric.

Assumptions Made:

    The text descriptions sufficiently describe the product categories.

    Categories are balanced in distribution (not necessarily true in real-world scenarios).

Limitations Include:

    The dataset is very small and synthetic, which might not represent real-world complexities and variances.
    
    Model simplicity might not capture complex patterns in data, thus may underperform on more diverse and large datasets.

Instructions:

- Run detect.py. The code will execute some automated test cases, after which you can input your own statements.

- To view evaluation metrics, run eval.py.

- To train with a new dataset, install the dataset as a CSV file and ensure it has the required format for columns. Then, place it in the assessment folder. Afterward, replace 'generated_data1.csv' with your dataset in line 46 of eval.py.

- You can generate a larger dataset using generate_train_data.py. It will create new statements with words replaced by their synonyms. You can adjust the size by modifying the augmentation_factor on line 65.

Screenshots:

![Screenshot 1](Screenshots/detect.png)
![Screenshot 2](Screenshots/eval.png)
![Screenshot 3](Screenshots/files.png)
![Screenshot 4](Screenshots/Generate_Data.png)
![Screenshot 5](Screenshots/train.png)


