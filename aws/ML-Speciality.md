# AWS Machine Learning & SageMaker – Comprehensive “All-in-One” Cheatsheet

This document merges all the **deep research** and **original SageMaker cheatsheet** content. It covers:

1. **General Machine Learning Concepts** (data imputation, class imbalance, transformations, encoding, feature engineering, evaluation metrics)
2. **AWS SageMaker Built-In Algorithms** (Linear Learner, XGBoost, k-Means, k-NN, Seq2Seq, DeepAR, BlazingText, Object2Vec, etc.)
3. **Additional AWS ML Services** (Comprehend, Rekognition, Transcribe, etc.)
4. **Key AWS ML Implementation Details** (SageMaker features, security, deployment, monitoring)
5. **Exam-Specific Tips** relevant to the AWS Certified Machine Learning – Specialty exam.

---

## 1. General ML Concepts

### 1.1 Data Imputation
- **Replace missing values** in features (columns).
- **Univariate**:
  - Mean or Median (median more robust to outliers).
  - Mode for categorical.
- **Multivariate**:
  - KNN Imputer: use neighbors to fill missing values.
  - Iterative Imputer (MICE): iteratively regress each missing feature based on others.

### 1.2 Class Imbalance
- **Downsampling + Upweighting**: cut majority class size, then apply higher weight to that class to compensate.
- **SMOTE**: oversample minority class by synthesizing new instances.
- **Undersampling**: removing majority class records (info loss risk).

### 1.3 Scaling & Transformation
- **Normalization (Min-Max)**: scale to [0,1] (affected by outliers).
- **Standardization (Z-score)**: center to mean=0, std=1 (less outlier impact).
- **Transformation**: 
  - Log, sqrt, etc. to reduce skew. 
  - For negative skew, reflect + transform.

### 1.4 Binning
- Convert continuous to discrete bins.
- **Unsupervised**: equal-width/frequency.
- **Supervised**: decision tree–style splits (entropy/gini).

### 1.5 Encoding Categorical Features
- **Label Encoding**: map categories to integers (nominal).
- **Ordinal Encoding**: integers that preserve rank.
- **One-Hot Encoding**: separate binary columns for each category.
- **Frequency Encoding**: map each category → its frequency.
- **Binary Encoding**: reduce high-cardinality (category → binary code).
- **Target Mean Encoding**: encode category → target average (watch for leakage).

### 1.6 Dimensionality Reduction
- **Feature selection**: remove unneeded or low-importance features.
- **PCA/SVD/Factor Analysis**: linear subspace reduction; can be sensitive to outliers.
- **t-SNE, UMAP**: non-linear manifold methods for visualization.
- **Autoencoders**: neural net–based compression.

### 1.7 Multi-Collinearity
- Detect with **Variance Inflation Factor (VIF)** or correlation matrix.
- Mitigate by removing correlated features or applying PCA/PLS.

---

## 2. Model Evaluation

### 2.1 Classification Metrics
- **Accuracy** = (TP + TN) / (TP+TN+FP+FN).
- **Precision** = TP / (TP+FP) (focus if false positives matter).
- **Recall** = TP / (TP+FN) (focus if false negatives matter).
- **F1-Score** = 2 * (Precision * Recall) / (Precision + Recall).
- **Specificity** = TN / (TN+FP).
- **ROC Curve & AUC**: TPR vs FPR. AUC=1 is perfect, 0.5 is random.

### 2.2 Regression Metrics
- **MSE, RMSE**: average of squared errors (RMSE in same units as target).
- **MAE**: mean absolute error, robust to outliers relative to MSE.
- **R²** = 1 - (SS_res / SS_tot); measures variance explained.

### 2.3 CNN/RNN Quick Notes
- **CNN**: conv layers + pooling → flatten → dense → softmax. Great for images.
- **RNN**: handle sequences. LSTM/GRU mitigate vanishing gradients.

### 2.4 Ensemble Methods
- **Bagging**: parallel training on bootstrapped subsets (e.g., Random Forest).
- **Boosting**: sequential training focusing on previous errors (e.g., XGBoost).

### 2.5 Neural Net Tuning
- **Activations**: ReLU or variants for hidden layers; softmax or sigmoid outputs.
- **Regularization**: dropout, L1/L2, early stopping.
- **Batch Size**: small batch → more generalization, large batch → faster but risk local minima.

---

## 3. SageMaker Essentials

**Amazon SageMaker** is a fully-managed service for ML at scale:
- Hosted notebooks (SageMaker Studio).
- Training jobs (built-in or custom containers).
- Hyperparameter tuning jobs.
- Endpoint deployment (scalable, multi-AZ).
- Batch Transform for offline inference.
- Debugger, Model Monitor, Clarify for bias/explainability, etc.

---

## 4. SageMaker Built-In Algorithms

This section goes in-depth on each built-in algorithm: **Linear Learner, XGBoost, k-Means, k-NN, Seq2Seq, DeepAR, BlazingText, Object2Vec, Object Detection, Image Classification, Semantic Segmentation, Random Cut Forest (RCF), Factorization Machines, Latent Dirichlet Allocation (LDA), Neural Topic Model (NTM), IP Insights**.

### 4.1 Linear Learner
- **Use Case**: regression or classification (binary/multiclass).
- **Input**: CSV or RecordIO-protobuf (Float32), no header, target in first column.
- **Training**:
  - Stochastic Gradient Descent (SGD) with variants (Adam, etc.).
  - Trains multiple “candidate” models, picks best by validation metric.
- **Regularization**: L1 (`l1`), L2 (`wd`).
- **Key Hyperparams**: `learning_rate`, `mini_batch_size`, `balance_multiclass_weights`.
- **Instances**: CPU or single GPU. Multi-GPU rarely helps.

### 4.2 XGBoost
- **Use Case**: classification (binary/multiclass) or regression.
- **Input**: CSV, libsvm, Parquet, or RecordIO-protobuf.
- **Key Hyperparams**:
  - `eta` (learning rate),
  - `max_depth`, `subsample`, `colsample_bytree`,
  - `alpha` (L1), `lambda` (L2),
  - `scale_pos_weight` (handle class imbalance).
- **Instances**: CPU (multi-instance) or single GPU (`tree_method=gpu_hist`). Often memory-bound.

### 4.3 K-Means
- **Use Case**: unsupervised clustering.
- **Input**: CSV or RecordIO-protobuf.
- **Hyperparams**:
  - `k` (number of clusters),
  - `init_method` (kmeans++ or random),
  - `mini_batch_size`.
- **Instances**: CPU or single-GPU, can do multi-instance. Evaluate with “elbow method.”

### 4.4 K-Nearest Neighbors (k-NN)
- **Use Case**: classification or regression using nearest neighbors.
- **Input**: CSV or RecordIO-protobuf, target in first column.
- **Steps**: sampling → dimension reduction (fjlt/sign) → index building.
- **Hyperparams**: `k`, `sample_size`, `dimension_reduction_type`.
- **Instances**: CPU/GPU, memory-bound for large datasets.

### 4.5 Seq2Seq
- **Use Case**: machine translation, text summarization (Seq → Seq).
- **Input**: RecordIO-protobuf with integer tokens, plus vocabulary files.
- **Model**: RNN or CNN with attention. Single-instance training, but multi-GPUs allowed in that instance.
- **Hyperparams**: `batch_size`, `optimizer_type`, `learning_rate`, `num_layers_encoder/decoder`.
- **Instances**: GPU only for training. CPU or GPU for inference.

### 4.6 DeepAR
- **Use Case**: forecasting time series (handles multiple related time series).
- **Input**: JSON (Gzip) or Parquet. Each record has “start”, “target” array, optional “cat” or “dynamic_feat”.
- **Hyperparams**: `context_length`, `prediction_length`, `epochs`, `mini_batch_size`.
- **Instances**: CPU or GPU, single or multi-machine.

### 4.7 BlazingText
- **Modes**:
  1. **Word2Vec** (unsupervised embeddings: skip-gram, cbow),
  2. **Text Classification** (supervised).
- **Input**:
  - Word2Vec: text (one sentence/line),
  - Text classification: lines start with `__label__<label>`.
- **Hyperparams**:
  - `mode` = skipgram / cbow / batch_skipgram,
  - `vector_dim`, `window_size`, `learning_rate`, etc.
- **Instances**:
  - For cbow/skipgram: single instance (CPU or GPU).
  - For batch_skipgram: multi-CPU or single-GPU.
  - For text classification >2GB: likely GPU needed.

### 4.8 Object2Vec
- **Use Case**: general embedding of “objects” (user-item pairs, text-text pairs, etc.).
- **Input**:
  - JSON lines of pairs (two channels: enc0, enc1).
- **Architecture**:
  - Two encoders → comparator → feed-forward → output label or embed.
- **Hyperparams**:
  - `enc0_*`, `enc1_*` (e.g., max_seq_len, vocab_size, network), 
  - `dropout`, `batch_size`, `learning_rate`.
- **Instances**:
  - Single CPU/GPU only, multi-GPU not supported.

### 4.9 Object Detection
- **Use Case**: bounding box + class detection in images.
- **Architecture**: Single Shot Detector (SSD) with base CNN (VGG-16 or ResNet50).
- **Input**:
  - RecordIO-protobuf or image files + annotation JSON.
- **Hyperparams**: `mini_batch_size`, `learning_rate`, `optimizer`.
- **Instances**: GPU only for training (multi-GPU, multi-instance possible). CPU/GPU for inference.

### 4.10 Image Classification
- **Use Case**: entire image classification (no bounding boxes).
- **Architecture**: ResNet under the hood, can do transfer learning.
- **Input**: RecordIO (MXNet) or raw image + .lst mapping.  
- **Hyperparams**: `batch_size`, `learning_rate`, `optimizer` (sgd, adam).
- **Instances**: GPU recommended, multi-GPU or multi-instance possible.

### 4.11 Semantic Segmentation
- **Use Case**: pixel-level classification (mask).
- **Architecture**: FCN, PSP, or DeepLabV3 with ResNet 50/101 backbone.
- **Input**: images + label maps (or augmented manifest).  
- **Instances**: single GPU for training. CPU/GPU inference.

### 4.12 Random Cut Forest (RCF)
- **Use Case**: unsupervised anomaly detection (time series or general numeric).
- **Input**: CSV or RecordIO-protobuf. Optional test channel for anomaly labels.
- **Hyperparams**:
  - `num_trees`, `num_samples_per_tree`.
- **Instances**: CPU only, not GPU-accelerated.

### 4.13 Factorization Machines
- **Use Case**: classification/regression in sparse data contexts (recommendations).
- **Input**: RecordIO-protobuf float32.  
- **Hyperparams**:
  - e.g., `feature_dim`, `mini_batch_size`, factor init methods, etc.
- **Instances**: CPU or GPU (GPU helps if data is dense).

### 4.14 Latent Dirichlet Allocation (LDA)
- **Use Case**: unsupervised topic modeling (CPU-based).
- **Input**: RecordIO-protobuf or CSV with word frequencies.
- **Hyperparams**:
  - `num_topics`, `alpha0`.
- **Instances**: single CPU instance, not parallelizable.

### 4.15 Neural Topic Model (NTM)
- **Use Case**: unsupervised topic modeling via neural variational inference.
- **Input**: CSV or RecordIO-protobuf with word IDs/freq.
- **Hyperparams**: `num_topics`, `mini_batch_size`, `learning_rate`.
- **Instances**: CPU or GPU, single or multi.

### 4.16 IP Insights
- **Use Case**: unsupervised IP usage anomaly detection.
- **Input**: CSV with entity (user/account) and IP address. Optional validation channel.
- **Hyperparams**:
  - `num_entity_vectors` (≥ 2× unique entities),
  - `vector_dim`, `epochs`, etc.
- **Instances**: CPU/GPU, single or multi-instance.

---

## 5. Additional SageMaker Features

### 5.1 Data Channels: File vs Pipe Mode
- **File Mode**: entire dataset downloaded to local disk.
- **Pipe Mode**: stream from S3, more efficient for big data, saves disk usage.

### 5.2 Deployment & Inference
- **Hosted Endpoints** (real-time): auto-scaling, multi-AZ.  
- **Batch Transform**: large offline inference tasks, no persistent endpoint needed.
- **Multi-Model Endpoints**: serve multiple models from one endpoint (by specifying model key).

### 5.3 Hyperparameter Tuning
- SageMaker HPO uses **Bayesian optimization** or random/grid search.
- You define ranges, objective metric, concurrency.
- Early stopping can reduce cost if a run is clearly not promising.

### 5.4 Debugger & Model Monitor
- **Debugger**: real-time capture of training metrics/tensors → detect issues (vanishing grads, etc.).
- **Model Monitor**: capture input data at inference, compare with training baseline → detect drift in schema/distribution.

### 5.5 SageMaker Neo
- Compiles trained models for optimized inference on target hardware (IoT, Inf1, edge devices).

---

## 6. Other AWS AI/ML Services

These **high-level APIs** let you add AI features without building custom models:

- **Amazon Comprehend**: text analytics (entities, sentiment, language, key phrases, topic modeling).
- **Amazon Rekognition**: vision (object detection, face recognition, celeb detection, text in image/video).
- **Amazon Transcribe**: speech to text, custom vocab, speaker identification.
- **Amazon Translate**: machine translation, custom terminology.
- **Amazon Polly**: text to speech, multiple voices, SSML.
- **Amazon Forecast**: time-series forecasting (AutoML approach).
- **Amazon Personalize**: personalized recommendations (collaborative filtering).
- **Amazon Lex**: conversational chatbots (ASR + NLU).
- **Amazon Fraud Detector**: train custom fraud detection models from your data.
- **Amazon Kendra**: enterprise search with NLP.
- **Lookout for Metrics / Vision / Equipment**: specialized anomaly detection.

Use these when domain tasks match (NLP, CV, speech, translation, etc.) and you prefer an API-driven approach.

---

## 7. ML Implementation & Operations

### 7.1 Training & Optimization
- **Cross-Validation**: train/val/test split or k-fold. 
- **Spot Instances**: cheaper but can be interrupted (use checkpoints).
- **Learning Rate**: too high → diverge, too low → slow training. 
- **Regularization**: L1/L2, dropout, early stopping.

### 7.2 Security & Compliance
- **IAM**: least privilege roles for SageMaker (train, access S3).
- **Encryption**: SSE-S3 or SSE-KMS for data at rest; TLS for data in transit.
- **VPC**: run training/inference in private subnets, use VPC endpoints for S3.
- **CloudTrail/CloudWatch**: track API calls, logs, metrics.

### 7.3 Deployment Patterns
- **Real-time endpoints**: auto-scaling, multi-AZ. 
  - Use production variants to do canary or A/B tests.
- **Batch inference**: one-time or periodic large-scale scoring (no always-on endpoint).
- **Edge**: SageMaker Neo + IoT Greengrass for edge deployments.
- **MLOps**: integrate code with CI/CD pipelines (CodePipeline, CodeBuild, SageMaker Pipelines, Model Registry).

### 7.4 Monitoring & Model Drift
- **Model Monitor**: checks if incoming data distribution changes significantly from the training baseline. 
- **Retraining**: schedule or trigger-based (time-based, drift-based, or new data).
- **Metrics**: track latency, error rates, resource usage. 
- **A/B Testing**: deploy new model at partial traffic, compare performance, roll forward/back.

### 7.5 Cost Optimization
- Use **Spot training** for up to 90% savings, with checkpoints.
- Use appropriate instances (e.g., CPU vs GPU) for your workload. 
- For inference, auto-scale or pick smaller instance if throughput is low.
- Use **Elastic Inference** for partial GPU acceleration on a CPU endpoint (supported frameworks).
- Monitor usage with Cost Explorer.

---

## 8. Exam-Specific Tips

1. **Familiarize thoroughly with built-in SageMaker algorithms**: input data format, hyperparameters, recommended instance usage, typical use cases.
2. **Know the difference** between real-time endpoints vs batch transform, file mode vs pipe mode, CPU vs GPU.
3. **Common hyperparams**:
   - Linear Learner: `learning_rate`, `mini_batch_size`, `wd` (L2), `l1`.
   - XGBoost: `eta`, `max_depth`, `subsample`, `colsample_bytree`, `alpha`, `lambda`.
   - K-Means: `k`, `init_method`.
   - DeepAR: `context_length`, `prediction_length`.
   - Seq2Seq: `num_layers_encoder`, `num_layers_decoder`, `batch_size`.
4. **Monitoring & logging**: CloudWatch logs & metrics, CloudTrail for auditing, SageMaker Debugger for training insights.
5. **Security**: S3 encryption, VPC endpoints, private subnets, IAM roles for SageMaker.
6. **Scaling & cost**: 
   - For training, consider spot or multi-machine. 
   - For inference, consider auto-scaling endpoints or using cheaper CPU with Elastic Inference if feasible.
   - SageMaker Tuning Jobs can find best hyperparams more efficiently than manual guesswork.
7. **AWS AI Services** vs **Custom ML**: if the problem is covered by an AI service (e.g. text classification → Comprehend, translation → Translate), you might not need custom modeling. Otherwise, use SageMaker built-ins or custom code.
8. **Evaluation**: know classification metrics (precision/recall/F1/AUC) and regression metrics (MSE, MAE, R²). Watch for class imbalance solutions.
9. **Implementation**: you may get scenario questions about architecture design, data ingestion (Kinesis, S3, Glue), training (single vs distributed), deployment approach, or production A/B testing.
10. **Data Prep**: AWS Glue for ETL, EMR for big data, or sagemaker_spark integration. Ensure your data is in the right format for the built-in algo you choose.

---

**End of Consolidated Cheatsheet**  
