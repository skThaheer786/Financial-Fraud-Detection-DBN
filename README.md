# Financial Fraud Detection Using Deep Generative Models

## Course Information

**Course:** Deep Learning  
**Course Code:** MLA04  
**Course Outcome:** CO5  
**Assignment:** Financial Fraud Detection  

## 1. Problem Statement

The objective of this project is to implement and evaluate a deep generative model for financial transaction data. A Deep Belief Network (DBN) based on stacked Restricted Boltzmann Machines (RBMs) is used to learn hidden representations of financial transactions and generate synthetic transaction records.

## 2. Dataset

The project uses a credit-card financial transaction dataset containing numerical transaction features.

The dataset contains:

- 40,136 transactions
- 31 processed features before model transformation
- Transaction-related numerical attributes
- A fraud classification attribute in the original dataset

The target/class column is excluded during unsupervised generative modeling.

## 3. Model Used

A Deep Belief Network (DBN) based on two Restricted Boltzmann Machines was implemented.

### Architecture

Input features: 30

First RBM:
- Visible units: 30
- Hidden units: 64

Second RBM:
- Input: 64
- Hidden units: 32

The learned hidden representation is then used to generate synthetic transaction data.

## 4. Generative Process

The model learns hidden representations from the training transactions.

The generation process consists of:

1. Loading the financial transaction dataset.
2. Removing unnecessary columns.
3. Handling missing values.
4. Separating features and target.
5. Splitting the dataset into training and testing data.
6. Converting the training data into PyTorch tensors.
7. Training the first RBM.
8. Extracting the hidden representation.
9. Training the second RBM.
10. Generating synthetic transaction samples.
11. Converting generated samples into a Pandas DataFrame.
12. Evaluating the generated data.

## 5. Generated Data

The trained DBN generated:

- 1,000 synthetic transactions
- 1,000 unique transactions
- 100% measured diversity

The generated data was saved as:

`synthetic_financial_transactions.csv`

## 6. Evaluation Results

| Metric | Result |
|---|---:|
| Synthetic transactions | 1,000 |
| Unique transactions | 1,000 |
| Diversity | 100% |
| Mean absolute difference | 3.355115 |
| Correlation difference | 0.269258 |
| Missing values | 0 |
| Infinite values | 0 |

## 7. Visualizations

The project includes visual evaluation of the generated data.

### Transaction Amount Distribution

The original transaction amount distribution is highly right-skewed. The synthetic distribution does not perfectly reproduce the original distribution, indicating that further model optimization is required.

### Correlation Matrix

The synthetic correlation matrix shows relationships between generated features. Some feature dependencies are captured, although the correlation structure does not perfectly match the original dataset.

## 8. Limitations

The current model has several limitations:

- Synthetic transaction amounts do not perfectly match the original distribution.
- Correlation relationships are only partially preserved.
- RBM training can be computationally expensive.
- The generated data may not represent rare financial fraud patterns accurately.
- More training epochs and hyperparameter optimization may improve the results.

## 9. Possible Improvements

Future improvements include:

- Increasing the number of training epochs.
- Optimizing learning rate and batch size.
- Increasing or tuning the number of hidden units.
- Using improved Gibbs sampling.
- Applying conditional generation for fraud/non-fraud classes.
- Comparing DBN with GANs, VAEs, or other generative models.
- Using additional statistical similarity metrics.

## 10. Reflection

This project helped in understanding deep generative models and their application to financial transaction data. The main design decision was to use a DBN consisting of stacked RBMs because RBMs can learn hidden representations and dependencies among input features.

A major challenge was generating synthetic data that accurately preserves the original statistical distribution. The experiment demonstrated that producing unique synthetic records is possible, but maintaining realistic feature distributions and relationships requires further optimization.

## 11. Conclusion

The DBN-based generative model successfully generated 1,000 unique synthetic financial transactions without missing or infinite values. The evaluation demonstrated high sample diversity and partial preservation of the statistical characteristics of the original dataset.

However, the difference in transaction amount distribution and feature correlations indicates that the model can be improved. Overall, the project demonstrates the application of deep generative learning for synthetic financial transaction generation.
