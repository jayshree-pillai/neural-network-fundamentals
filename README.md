🧠 Neural Network from Scratch — Iris Classification
This project implements a 2-layer neural network completely from scratch using NumPy, without relying on any deep learning libraries. The model is trained to classify Iris flower species based on their petal and sepal measurements.

🚀 What’s Inside
Manual forward pass: ReLU + Softmax activations

Manual backward pass: Full derivation + vectorized gradients

Custom loss: Binary and categorical cross-entropy

Gradient descent updates

Modular class: FlexibleNN(input_dim, hidden_dim, output_dim)

Comparison: Equivalent TensorFlow model for validation

🧪 Dataset
Iris Dataset from sklearn.datasets

3 output classes: Setosa, Versicolor, Virginica

Inputs: 4 numerical features (lengths & widths)

Train/test split with stratified sampling

Standardized using StandardScaler

✅ Results
Achieved ~95%+ accuracy on test data using the scratch model
Matched performance with TensorFlow benchmark

🧩 Key Learnings
Mechanics of forward and backward propagation
Role of activation derivatives in gradient flow
Impact of initialization, loss scaling, and learning rate

