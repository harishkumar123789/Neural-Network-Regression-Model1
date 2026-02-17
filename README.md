# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.
## Neural Network Model

![image](https://github.com/user-attachments/assets/84093ee0-48a5-4bd2-b78d-5d8ee258d189)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: HARISH KUMAR S
### Register Number:212224230091
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x)) 
    x=self.relu(self.fc2(x))
    x=self.fc3(x)  
    return x


# Initialize the Model, Loss Function, and Optimizer

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
    



```
## Dataset Information

<img width="306" height="461" alt="image" src="https://github.com/user-attachments/assets/986d0064-b072-4d94-a772-03b0b5ffeee2" />


## OUTPUT

### Training Loss Vs Iteration Plot
<img width="767" height="584" alt="image" src="https://github.com/user-attachments/assets/3a4f2da8-5034-478c-97e0-6a7d21d56f65" />



### New Sample Data Prediction
<img width="1035" height="156" alt="image" src="https://github.com/user-attachments/assets/b2d2f00f-815a-4912-866b-4f7445fd7bfa" />


## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
