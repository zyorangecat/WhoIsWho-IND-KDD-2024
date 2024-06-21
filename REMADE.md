# WhoIsWho-IND-KDD-2024 Rank4

## Tree modeling process&train

### Prerequisites
```python
pip install -r requirements.txt
```

### Data processing

1. **Load data**
   - Read training data, including authors and their correctly and incorrectly assigned papers.
   - Read detailed information of all papers, including title, author, abstract, keywords, conference or journal, and publication year.
   - Read test data, including authors and all their papers that need to be verified.
2. **Convert to DataFrame**

   - Convert the dictionary format of training data to Pandas DataFrame format for subsequent processing.
   - Process correctly assigned papers and incorrectly assigned papers separately, and add labels (the label of correctly assigned papers is 1, and the label of incorrectly assigned papers is 0).
   - Merge the processed data to form the final training data set.
3. **Process test data**
   - Convert test data to DataFrame format and extract all paper information of each author.
4. **Process paper details**
   - Convert the detailed information dictionary of the paper to DataFrame format to facilitate subsequent feature engineering.

### Feature engineering

Feature engineering is the core part of this project. It uses a variety of techniques to extract useful features from data to improve the performance of the model. We used the following feature engineering techniques:

1. **Keyword processing**

   - Extract a list of common keywords from the keywords of the paper.

   - Count the number of occurrences of these keywords in each paper and generate keyword features.

2. **Author information processing**

   - Parse the author information in each paper and count the number of papers for each author.

   - Generate matching features between authors and papers.

3. **Text processing**

   - Perform text processing on the title and abstract of the paper, and extract text features using multiple methods [using idf and w2v as examples].

   - Use the TF-IDF (term frequency-inverse document frequency) method to extract text features.

   - Generate text embedding using the Word2Vec model.

   - Calculate the similarity between the title and abstract and generate text similarity features.

4. **Paper publication information processing**

   - Extract the year of paper publication and generate year features.

   - Extract the conference or journal information where the paper was published and generate classification features.

5. **Other features**

   - Combine the above features to generate interactive features (such as the interaction between authors and keywords, the interaction between the publication year and keywords, etc.).

   - Generate embedding features based on paper ID, and embed paper ID through Word2Vec model.

### Model training

1. **Data preparation**
   - Use stratified K-fold cross-validation method to divide the training data into training set and validation set.

2. **Model selection**
   - Select LightGBM and XGBoost models for training respectively.

   - Use early stopping and logging functions to optimize the model training process.

3. **Model training and validation**

   - Train LightGBM and XGBoost models on the training set, and evaluate the models on the validation set.

   - Use ROC AUC as the model evaluation indicator to select the best model.

4. **Model fusion**
   - Fusion the prediction results of LightGBM and XGBoost models to improve the stability and generalization ability of the model.

5. **Model prediction**
   - Use the fused model to predict the test data and output the prediction results.

## NN modeling process&train

1. **CHatGLM3-32k**
    
    - Use the official base and modify some parameters (MAX_SOURCE_LEN, LR, EPOCH)
   - Device: 8*A100
   ```python
      cd ChatGLM3
      train：bash train.sh
      infer：bash test.sh
    ```
    
2. **Llama3-6b**
    - Modify according to the official base and adjust the DataSet and DataCollator to align the input and output of Llama
    - Device 8*A100
   ```python
      cd llama3
      train：bash train.sh
      infer：bash test.sh
   ```

3. **GCN**
    - From the official
   ```python
      cd GCN
      python encoding.py
      python build_graph.py --author_dir /data/laiguibin/LLMs/incorrect_assignment_detection/data/IND-WhoIsWho/train_author.json --save_dir /data/laiguibin/LLMs/incorrect_assignment_detection/data/IND-WhoIsWho/train.pkl
      python build_graph.py
      python train.py
   ```

## Model fusion

We perform weighted fusion on the output results of the above models. The weight consideration mainly comes from the difference between online scores and model estimates.
   ```python
    cd LGBM
    python lgb_xgb.py 
    python final.py # Model fusion
    
   ```
## Parameter&Device Description

**parameter**：8,000,000,000

**total video memory(GB)**：640

**Device**: CPU 64C.256G / GPU 8*A100

