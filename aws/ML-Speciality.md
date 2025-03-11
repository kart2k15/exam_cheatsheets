# **Master AWS Machine Learning & SageMaker Cheatsheet with Tables**

Below is a consolidated reference that merges:
- **General ML concepts** (data imputation, scaling, class imbalance, transformations, etc.)
- **Key SageMaker built-in algorithms** (with detailed **tables** summarizing input format, training/inference instance support, support for transfer/incremental learning, etc.)
- **SageMaker operational features** (Hyperparameter Tuning, Debugger, Model Monitor)
- **Other AWS AI services** (Comprehend, Rekognition, Transcribe, etc.)
- **Practical exam tips** (which instance, which format, how to handle data drift, etc.)

---

## 1. General ML Concepts

### 1.1 Data Imputation & Missing Values
- **Univariate** strategies:
  - **Mean** or **Median** for numeric (median is robust to outliers).
  - **Mode** for categorical.
- **Multivariate** strategies:
  - **KNN Imputer**: fill missing based on feature similarity among neighbors.
  - **Iterative Imputer / MICE**: iteratively fits regressors on known values to predict missing ones.

### 1.2 Handling Class Imbalance
- **Downsampling + Upweighting**: trim majority class, then weight it proportionally.
- **SMOTE**: synthesize minority-class examples (oversampling).
- **Undersampling**: remove many majority examples (simple but can lose data).

### 1.3 Scaling & Normalization
- **Normalization (Min-Max)**: transform to [0,1] (sensitive to outliers).
- **Standardization (Z-score)**: mean=0, std=1. Less outlier-sensitive, but distribution shape remains skewed if originally skewed.
- **Transformations (log, sqrt)**: reduce skew. For negative skew, reflect + log.

### 1.4 Feature Binning & Encoding
- **Binning**: 
  - *Unsupervised*: equal-width/frequency. 
  - *Supervised*: split based on Gini/information gain (decision-tree style).
- **Encoding**:
  - Label/Ordinal (integer mapping),
  - One-Hot (dummy columns),
  - Frequency/Binary encoding (reduce dimensionality),
  - Target mean encoding (risk of leakage, use with care).

### 1.5 Dimensionality Reduction
- **PCA** / **SVD**: linear projections (sensitive to outliers).
- **t-SNE** / **MDS** / **UMAP**: non-linear manifold approaches (often for visualization).
- **Autoencoders**: neural compression.

### 1.6 Model Evaluation Metrics
- **Classification**:
  - Accuracy = (TP+TN)/Total
  - Precision = TP/(TP+FP)
  - Recall (Sensitivity) = TP/(TP+FN)
  - F1 = harmonic mean of precision & recall
  - ROC-AUC: area under ROC curve; 1=perfect, 0.5=random
- **Regression**:
  - MSE / RMSE, MAE, R²
- **Imbalanced Data**: consider F1, PR-AUC, or AUC-ROC with caution.

### 1.7 Neural Networks & Ensembles
- **CNN** for images; **RNN** (LSTM/GRU) for sequences.
- **Bagging** (Random Forest) vs **Boosting** (XGBoost).  
- **Regularization**: dropout, early stopping, L1/L2.

---

## 2. SageMaker Built-In Algorithms Overview

The table below **summarizes** each built-in algorithm’s key attributes:  
- **Type**: Supervised vs Unsupervised  
- **Use Cases**: classification, regression, clustering, etc.  
- **Input Format**: CSV, RecordIO-protobuf, JSON, etc.  
- **Transfer/Incremental**: whether the algo supports transfer learning or incremental training.  
- **Training Instances**: CPU/GPU usage, multi-machine support, etc.  
- **Inference Instances**: typical CPU/GPU usage for deployment.  
- **Notes**: special mention of data channel modes (File vs Pipe), or any unique constraints.

> **Legend**:  
> - “CSV” = first column is label (if supervised), no header.  
> - “RecordIO” = RecordIO-protobuf, often float32 for built-ins.  
> - “JSON” = lines of JSON (DeepAR, IP Insights, etc.).  
> - “Yes” or “No” in *Transfer/Incremental* might reflect partial or direct support.  
> - For multi-GPU vs multi-machine, “Yes” means it can scale horizontally or can benefit from multiple GPU training, as indicated by AWS docs or typical usage.

| **Algorithm**         | **Type**                       | **Use Cases**                                        | **Input Formats**                                                  | **Transfer / Incremental?**       | **Training Instances**                                                                   | **Inference Instances**                         | **Notes**                                                                                                                                                                           |
|-----------------------|--------------------------------|-------------------------------------------------------|--------------------------------------------------------------------|------------------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Linear Learner**    | Supervised (Classif/Regress)   | General classification/regression                     | - CSV (label in col 1, no header)<br>- RecordIO (float32)           | **No**                           | - CPU or single GPU<br>- Multi-CPU possible<br>- Multi-GPU not beneficial               | - CPU or GPU                                    | - Can parallel-train multiple models w/ diff loss<br>- L1 (`l1`) & L2 (`wd`) regularization<br>- File or Pipe mode <br>- Great for tabular data, smaller to medium scale              |
| **XGBoost**           | Supervised (Classif/Regress)   | Wide range (binary, multiclass, regression)           | - CSV, libsvm, Parquet<br>- RecordIO (optional)                    | **No**                           | - Multi-CPU (distributed)<br>- Single-GPU (`tree_method=gpu_hist`)<br>- Typically memory-bound | - CPU or GPU                                    | - Very popular gradient boosting library<br>- Key HP: `eta`, `max_depth`, `subsample`, `alpha`, `lambda`<br>- Good for tabular data, can handle large scale                          |
| **k-Means**           | Unsupervised (Clustering)      | Clustering / grouping data                            | - CSV or RecordIO                                                  | **No**                           | - CPU or single GPU<br>- Multi-instance CPU possible                                     | - CPU or GPU                                    | - “init_method” = kmeans++ or random<br>- `k` clusters, `mini_batch_size`<br>- Good for segmentation tasks; pipe mode for large data                                                |
| **k-NN**              | Supervised (Classif/Regress)   | k-Nearest Neighbors                                   | - CSV (label col 1)<br>- RecordIO                                   | **No**                           | - CPU or GPU single instance<br>- Typically memory-bound<br>- Not usually multi-machine  | - CPU or GPU                                    | - 3 steps: sampling → dimension reduction (fjlt/sign) → index<br>- `k` and `sample_size` are main hyperparams                                                                       |
| **Seq2Seq**           | Supervised (Sequence → Seq)    | Machine translation, summarization, text gen          | - RecordIO (integer tokens)<br>+ vocab files in S3                  | **Yes** (pretrained models exist) | - **Single instance GPU** only<br>- But can use multi-GPU on that instance.<br>- No multi-machine | - CPU or GPU                                    | - RNN or CNN with attention<br>- Evaluate with BLEU, perplexity<br>- Must provide vocab & tokenized data<br>- Good for language tasks                                               |
| **DeepAR**            | Supervised (Forecasting)       | Time-series forecasting (univariate)                  | - JSON (Gzip) or Parquet lines<br>with start timestamp + target arr | **No**                           | - CPU or GPU<br>- Single or multi-machine                                                | - CPU or GPU                                    | - Learns global model across many related time series<br>- Key HP: `context_length`, `prediction_length`, `epochs`<br>- Good for inventory, sales, metrics forecasting               |
| **BlazingText**       | Un/supervised (Word2Vec or TC) | 1) word2vec embeddings<br>2) supervised text classify | - word2vec: text lines<br>- classification: lines w/ `__label__`    | **No**                           | - skipgram/cbow: single instance CPU/GPU<br>- batch_skipgram: multi-CPU or single GPU    | - CPU or GPU (text classification)              | - For word2vec: mode=skipgram, cbow, or batch_skipgram<br>- For text classification: n-grams, adjustable vector_dim<br>- Large data → better on GPU or multi-CPU (batch_skip)        |
| **Object2Vec**        | Supervised (pairs)             | Pairwise embedding (user-item, Q&A pairs, etc.)       | - JSON lines (two channels of tokens)                               | **No**                           | - Single CPU or GPU only<br>- No multi-machine                                          | - CPU or GPU                                    | - Two encoders, then comparator layer<br>- Good for similarity, recommendations, text matching<br>- Key HP: enc0/enc1 vocab, dropout, learning_rate                                |
| **Object Detection**  | Supervised (Vision)            | Bounding boxes + classes for objects in image         | - RecordIO-protobuf or images + JSON for annotations               | **Yes** (transfer from pretrained) | - **GPU only** for training<br>- Multi-GPU + multi-instance possible                     | - CPU or GPU                                    | - Single Shot Detector w/ base CNN (ResNet/VGG)<br>- Transfer learning from pretrained models<br>- Key HP: batch size, learning_rate, etc.<br>- Augmentation (flip, jitter)          |
| **Image Classification** | Supervised (Vision)        | Image-level classification (no bounding boxes)        | - RecordIO (MXNet) or raw images + .lst                             | **Yes** (transfer from pretrained) | - GPU recommended<br>- Supports multi-GPU & multi-machine                                | - CPU or GPU                                    | - ResNet-based CNN<br>- Transfer learning from pretrained ResNet<br>- batch_size, learning_rate, etc.                                                                             |
| **Semantic Segmentation** | Supervised (Vision)       | Pixel-level image segmentation                        | - Images + label maps<br>- Possibly augmented manifest for pipe     | **Yes** (transfer from pretrained) | - **Single GPU** only (no multi-GPU)<br>- Single instance CPU not supported for training (requires GPU) | - CPU or GPU                                    | - FCN, PSP, or DeepLabV3 w/ ResNet 50/101<br>- Transfer learning from ImageNet<br>- Produces mask per pixel                                                                        |
| **Random Cut Forest** | Unsupervised (Anomaly)         | Detect outliers in numeric/time-series                | - CSV or RecordIO                                                   | **No**                           | - **CPU only** (no GPU acceleration)<br>- Single instance recommended                     | - CPU                                           | - num_trees, num_samples_per_tree are key<br>- Good for streaming anomalies, integrated w/ Kinesis in doc examples                                                                  |
| **Factorization Machines** | Supervised (Classif/Reg) | Sparse data, recommenders, click prediction           | - RecordIO-protobuf (float32 only)                                  | **No**                           | - CPU or GPU, typically single instance                                                 | - CPU or GPU                                    | - Learns factorized interactions among features<br>- Good for high-dimensional sparse data (recommendations)<br>- Key HP: feature_dim, factor init, etc.                            |
| **Latent Dirichlet Allocation (LDA)** | Unsupervised (Topic Model) | Topic clustering in docs                       | - RecordIO or CSV w/ doc word freq                                   | **No**                           | - **CPU only**, single instance (not parallelizable)                                     | - CPU                                           | - Traditional LDA approach, can do test channel for perplexity<br>- alpha0 hyperparam controls topic mixtures                                                                      |
| **Neural Topic Model (NTM)** | Unsupervised (Topic Model) | Neural variational approach to topics            | - CSV or RecordIO w/ word freq vectors                              | **No**                           | - CPU or GPU<br>- Single or multi instance                                              | - CPU or GPU                                    | - Similar to LDA but uses autoencoder-like approach<br>- Key HP: num_topics, learning_rate, mini_batch_size                                                                         |
| **IP Insights**       | Unsupervised (Anomaly/Fraud)   | Detect unusual IP usage for entities                 | - CSV: entity, IP address, optional validation channel               | **No**                           | - CPU or GPU<br>- Single or multi instance                                              | - CPU or GPU                                    | - Embeds IP + entity in latent space, flags anomalies<br>- Key HP: num_entity_vectors, vector_dim, epochs                                                                          |

> **Key Transfer-Learning** Built-Ins:
> - **Object Detection** (transfer from pretrained ResNet/VGG),
> - **Image Classification** (transfer from pretrained ResNet),
> - **Semantic Segmentation** (transfer from pretrained ResNet),
> - **Seq2Seq** can optionally start from a pretrained model for translation tasks.

---

## 3. Additional SageMaker Features

### 3.1 Hyperparameter Tuning
- **Automatic Model Tuning** with Bayesian optimization or random search.
- You define hyperparam ranges + objective metric.
- Tuning job spawns multiple training jobs, picking new combos based on previous results.

### 3.2 Debugger (SMDebug)
- Captures training metrics (weights, gradients) at intervals.
- Has built-in or custom **rules** to detect vanishing gradients, overfitting, etc.
- Integrates with CloudWatch events for real-time alerts.

### 3.3 Model Monitor
- Monitors real-time inference requests/responses at endpoints.
- Compares distribution to baseline (from training set).
- Alerts if drift or anomalies appear in feature values.

### 3.4 Deployment: Real-Time vs Batch
- **Real-time endpoints**: synchronous inference over HTTPS, can auto-scale, multi-AZ. 
- **Batch transform**: offline scoring for large S3 datasets, no persistent endpoint cost.

### 3.5 SageMaker Neo
- **Compile** trained models for optimized edge/hardware deployment (Inf1, GPUs, ARM).

---

## 4. Other AWS AI Services

- **Amazon Comprehend**: NLP tasks (sentiment, entities, key phrases).
- **Amazon Rekognition**: image/video detection, face analysis, custom labels.
- **Amazon Transcribe**: speech-to-text (batch or streaming).
- **Amazon Translate**: machine translation (supports custom terminologies).
- **Amazon Polly**: text-to-speech with diverse voices.
- **Amazon Forecast**: time-series forecasting (automated).
- **Amazon Personalize**: recommendation systems (collaborative filtering).
- **Amazon Lex**: chatbot (ASR + NLU).
- **Fraud Detector**: custom fraud detection from your event data.
- **Kendra**: enterprise search with ML-based Q&A.

These are **API-driven** solutions, minimal ML knowledge required.

---

## 5. Operational Best Practices & Exam Tips

1. **Data Formats**: 
   - Many built-ins expect CSV with label in column 1.  
   - Or RecordIO-protobuf for performance.  
   - Some use JSON (DeepAR, IP Insights).  
   - Ensure correct shape (float32, integer tokens, etc.) or the job fails.

2. **Security**: 
   - S3 encryption (SSE-S3 or SSE-KMS),  
   - VPC-only training/endpoints (no public internet),  
   - IAM roles with least privilege (allow SageMaker to read input data, write model artifacts, etc.).

3. **Instance Selection**:
   - **GPU** needed for deep neural nets (Object Detection, Segmentation, Seq2Seq).  
   - Some are CPU-only (LDA, RCF).  
   - Check if multi-machine or multi-GPU is actually beneficial (e.g., XGBoost or object detection scale well, but Linear Learner multi-GPU is not beneficial).
   - Use **Spot** for cheaper training (checkpointing recommended).

4. **Monitoring & Logging**:
   - CloudWatch for logs & metrics (e.g., CPU/memory, error rates).
   - CloudTrail for API calls (auditing).
   - Model Monitor for data drift in production.

5. **When to Use Built-In vs Custom**:
   - If the problem is a standard ML pattern (classification, regression, clustering, object detection, forecasting, etc.) the built-ins can be quick & optimized.
   - Otherwise, bring your own container or framework (PyTorch, TensorFlow, SKLearn, etc.).

6. **Hyperparameter Tuning**:
   - Provide param ranges, define objective metric (like `validation:accuracy`).
   - Use log scales for wide dynamic ranges (learning_rate, regularization).
   - SageMaker supports early stopping for poor runs.

7. **Deployment Approaches**:
   - Real-time endpoints for low-latency scoring, auto-scale with production variants for A/B tests.
   - Batch transform for offline scoring on big datasets.
   - Multi-model endpoint if you serve many smaller models.

8. **Exam-Specific**:
   - Expect scenario questions about picking the right built-in algo & correct input format.
   - Distinguish real-time vs batch vs AI service usage.
   - Remember metric usage (Precision/Recall, AUC, MSE, etc.) and data security.

---

**End of Enhanced Cheatsheet**  
Use the tables for quick reference on each built-in SageMaker algorithm’s input requirements, transfer learning capability, and instance usage. Combine these best practices with broader AWS knowledge (S3, IAM, VPC, Glue, Kinesis, etc.) to handle typical end-to-end ML scenarios in the AWS Certified Machine Learning – Specialty exam.
