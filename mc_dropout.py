from sklearn.datasets import make_blobs, make_circles, load_iris
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import math
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from tqdm import trange
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h1 style = 'text-align: center;'>MC DropOut</h1>", unsafe_allow_html = True)

torch.manual_seed(42)

# Model
class classification2(nn.Module):
  def __init__(self, input_layers, output_layers, hidden_layers_1 = 16, hidden_layers_2=16):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(input_layers, hidden_layers_1),
        nn.ReLU(),
        # nn.Dropout(p=0.2),
        nn.Linear(hidden_layers_1, hidden_layers_2),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_layers_2, output_layers),
        # nn.Softmax(dim=1)
    )

  def forward(self, x : torch.Tensor) -> torch.Tensor:
      output = self.model(x)
      return output

class classification_iris(nn.Module):
  def __init__(self, input_layers, output_layers, hidden_layers_1 = 16, hidden_layers_2=16):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(input_layers, hidden_layers_1),
        nn.ReLU(),
        # nn.Dropout(p=0.2),
        nn.Linear(hidden_layers_1, hidden_layers_2),
        nn.ReLU(),
        nn.Linear(hidden_layers_1, hidden_layers_2),
        nn.ReLU(),
        nn.Linear(hidden_layers_1, hidden_layers_2),
        nn.ReLU(),
        nn.Linear(hidden_layers_1, hidden_layers_2),
        nn.ReLU(),
        nn.Linear(hidden_layers_1, hidden_layers_2),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_layers_1, hidden_layers_2),
        nn.ReLU(),
        nn.Linear(hidden_layers_1, hidden_layers_2),
        nn.ReLU(),
        nn.Linear(hidden_layers_2, output_layers),
        # nn.Softmax(dim=1)
    )

  def forward(self, x : torch.Tensor) -> torch.Tensor:
      output = self.model(x)
      return output

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.
    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = (torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, alpha = 1)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.6)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def plot_mean_contour(model, x, iterations, x_val = None, y_val = None, z_val =None):
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    stack = []
    for iteration in trange(iterations):
        enable_dropout(model)
        with torch.inference_mode():
            val_logits = model(X_to_pred_on)
            val_pred = torch.round(torch.sigmoid(val_logits))
            stack.append(np.array(val_pred))

    stack = np.array(stack)
    stack = stack.T

    mean = stack.mean(axis = 2)
    mean = mean.reshape(xx.shape)

    plt.contourf(xx, yy, mean, cmap=plt.cm.RdYlBu)
    cbar = plt.colorbar(plt.contourf(xx, yy, mean, cmap=plt.cm.RdYlBu))
    if x_val is not None and y_val is not None and z_val is not None:
      plt.scatter(x_val, y_val, c=z_val, cmap="viridis", alpha=0.5)
    plt.title("Mean Boundary")
    plt.show()

def plot_mean_contour_iris(model, x, iterations, x_val = None, y_val = None, z_val =None):
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    stack = []
    for iteration in trange(iterations):
        enable_dropout(model)
        with torch.inference_mode():
            val_logits = model(X_to_pred_on)
            val_pred = torch.argmax(torch.softmax(val_logits, dim = 0), axis = 1)
            stack.append(np.array(val_pred))

    stack = np.array(stack)
    stack = stack.T

    mean_ = np.round(stack.mean(axis = 1))

    print(mean_, mean_.shape)
    mean_ = mean_.reshape(xx.shape)

    plt.contourf(xx, yy, mean_, cmap=plt.cm.RdYlBu)
    cbar = plt.colorbar(plt.contourf(xx, yy, mean_, cmap=plt.cm.RdYlBu))
    if x_val is not None and y_val is not None and z_val is not None:
      plt.scatter(x_val, y_val, c=z_val, cmap="viridis", alpha=0.5)
    plt.title("Mean Boundary")
    plt.show()

def plot_std_contour(model, x, iterations, x_val =None, y_val=None, z_val=None):
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    stack = []
    for iteration in trange(iterations):
        enable_dropout(model)
        with torch.inference_mode():
            val_logits = model(X_to_pred_on)
            val_pred = torch.round(torch.sigmoid(val_logits))
            stack.append(np.array(val_pred))

    stack = np.array(stack)
    stack = stack.T

    std = stack.std(axis = 2)
    std = std.reshape(xx.shape)

    plt.contourf(xx, yy, std, cmap=plt.cm.RdYlBu)
    cbar = plt.colorbar(plt.contourf(xx, yy,std, cmap=plt.cm.RdYlBu))
    if x_val is not None and y_val is not None:
      plt.scatter(x_val, y_val, c=z_val, cmap="viridis", alpha=0.5)
    plt.title("Standard Deviation")
    plt.show()

def plot_std_contour_iris(model, x, iterations, x_val = None, y_val = None, z_val =None):
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    stack = []
    for iteration in trange(iterations):
        enable_dropout(model)
        with torch.inference_mode():
            val_logits = model(X_to_pred_on)
            val_pred = torch.argmax(torch.softmax(val_logits, dim = 0), axis = 1)
            stack.append(np.array(val_pred))

    stack = np.array(stack)
    stack = stack.T

    std_ = (stack.std(axis = 1))
    # std_.shape
    std_ = std_.reshape(xx.shape)
    plt.contourf(xx, yy, std_, cmap=plt.cm.RdYlBu)
    cbar = plt.colorbar(plt.contourf(xx, yy, std_, cmap=plt.cm.RdYlBu))
    # if x_val is not None and y_val is not None and z_val is not None:
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", alpha=0.5)
    plt.title("Standard Deviation")
    plt.show()

# Progress bar status
txt1 = "Model is Training..."
txt2 = "Almost There..."
txt3 = "Done!"

# title_placeholder = st.empty()
# dataset_placeholder = st.empty()

if __name__=="__main__":
    with st.sidebar:
        st.markdown("<h1 style ='text-align: center;'>Menu</h1>", unsafe_allow_html = True)
        home =st.button("Home :house:", use_container_width = True)
        Circles = st.button("Circles", use_container_width = True)
        Blobs = st.button("Blobs", use_container_width = True)
        Iris = st.button("Iris Dataset", use_container_width = True)
        Generate = st.button("Generate", use_container_width = True, type = "primary")
        Train = st.button("Train", use_container_width = True, type = "primary")

    if "circle" not in st.session_state:
        st.session_state["circle"] = False
    if "blobs" not in st.session_state:
        st.session_state["blobs"] = False
    if "generate" not in st.session_state:
        st.session_state["generate"] = False
    if "train" not in st.session_state:
        st.session_state["train"] = False
    if "iris" not in st.session_state:
        st.session_state["iris"] = False
    if "home" not in st.session_state:
        st.session_state["home"] = False

    if Circles:
        st.session_state["circle"] = True
        st.session_state["blobs"] =False
        st.session_state["iris"] = False
        st.session_state["home"] = False
    if Blobs:
        st.session_state["blobs"] = True
        st.session_state["circle"] = False
        st.session_state["iris"] = False
        st.session_state["home"] = False
    if Generate:
        if st.session_state["circle"] or st.session_state["blobs"] or st.session_state["iris"]:
            st.session_state["generate"] = True
    if st.session_state["generate"]:
        if Train:
            st.session_state["train"] = True
    if Iris:
        st.session_state["iris"] = True
        st.session_state["circle"] = False
        st.session_state["blobs"] = False
        st.session_state["home"] = False
    if home:
        st.session_state["home"] = True
        st.session_state["iris"] = False
        st.session_state["circle"] = False
        st.session_state["blobs"] = False

    if st.session_state["home"]:
        st.header("Hello")

    if st.session_state["circle"] or st.session_state["blobs"]:
        if st.session_state["circle"]:
            samples = st.slider("Circles Samples", 0, 10000)
            iterations = st.slider("MCDropOut Iterations (Circle)", 0, 500)

            if st.session_state["generate"]:
                st.header("Circles Dataset")
                # Generate a synthetic dataset with 100 samples and noise=0.1
                X, y = make_circles(n_samples=samples, noise=0.1)

                # Plot the generated dataset
                plt.scatter(X[:, 0], X[:, 1], c=y)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Circles')
                plt.show()
                st.pyplot()

                train_ratio = 0.8
                val_ratio = 0.1
                test_ratio =0.1

                x = torch.Tensor(X)
                y = torch.Tensor(y)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_ratio + val_ratio)
                x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = val_ratio / (val_ratio + test_ratio))
                y_train = y_train.reshape((int(train_ratio * samples),1))
                y_val = y_val.reshape((int(val_ratio * samples),1))
                y_test = y_test.reshape((int(test_ratio * samples),1))

        if st.session_state["blobs"]:
            samples = st.slider("Blobs Samples", 0, 10000)
            iterations = st.slider("MCDropOut Iterations (Blobs)", 0, 500)
            
            if st.session_state["generate"]:
                st.header("Blobs Dataset")
                x, y = make_blobs(n_samples=samples, n_features=2, centers=2, random_state=42, )

                plt.scatter(x[:, 0], x[:, 1], c=y)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Blobs')
                plt.show()
                st.pyplot()

                train_ratio = 0.8
                val_ratio = 0.1
                test_ratio =0.1

                x = torch.Tensor(x)
                y = torch.Tensor(y)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_ratio + val_ratio)
                x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = val_ratio / (val_ratio + test_ratio))
                y_train = y_train.reshape((int(train_ratio * samples),1))
                y_val = y_val.reshape((int(val_ratio * samples),1))
                y_test = y_test.reshape((int(test_ratio * samples),1))

        if st.session_state["generate"]:
            if st.session_state["train"]:   
                model2 = classification2(2, 1)
                loss_fn = BCEWithLogitsLoss()
                opt = torch.optim.SGD(model2.parameters(), lr=0.05)

                epochs = 10000
                train_loss_model2 = []
                val_loss_model2 = []

                my_bar = st.progress(0, text = txt1)

                for epoch in trange(epochs):
                    model2.train()

                    logits = model2(x_train)
                    pred = torch.round(torch.sigmoid(logits))

                    loss = loss_fn(logits, y_train)
                    train_loss_model2.append(loss.detach().numpy())

                    loss.backward()

                    opt.step()

                    opt.zero_grad()

                    model2.eval()
                    
                    with torch.inference_mode():
                        val_logits = model2(x_val)
                        val_pred = torch.round(torch.sigmoid(val_logits))
                        val_loss_model2.append(loss_fn(model2(x_val), y_val).detach().numpy())
                    if int((epoch+1)*100/epochs)<80:
                        my_bar.progress(int((epoch+1)*100/epochs), text = txt1)
                    else:
                        if int((epoch+1)*100/epochs)<100:
                            my_bar.progress(int((epoch+1)*100/epochs), text = txt2)
                        else:
                            my_bar.progress(int((epoch+1)*100/epochs), text = txt3)

                st.markdown("<h2 style ='text-align: center;'>Results</h2>", unsafe_allow_html = True) 
                st.header("Trained Decision Boundary")    
                plot_decision_boundary(model2, x,y)
                st.pyplot()
                st.header("Mean Decision Boundary after MC DropOut")
                plot_mean_contour(model2, x, iterations)
                st.pyplot()
                # plot_mean_contour(model2, int(x[:,0].min())-1, int(x[:,0].max())+1, int(x[:,1].min())-1, int(x[:,1].max())+1, iterations)
                # st.pyplot()
                st.header("Standard Deviation after MC DropOut")
                plot_std_contour(model2, x, iterations)
                st.pyplot()
                
                st.session_state["circle"] = False
                st.session_state["blobs"] = False
                st.session_state["generate"] = False
                st.session_state["train"] = False

    if st.session_state["iris"]:
        iris_options = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        selected = st.multiselect("Select attributes: ", iris_options, max_selections = 2)
        iterations = st.slider("MCDropOut Iterations (Iris)", 0, 500)

        if st.session_state["generate"]:
            st.header("Iris Dataset")

            list_ = []
            if "Sepal Length" in selected:
                list_.append(0)
            if "Sepal Width" in selected:
                list_.append(1)
            if "Petal Length" in selected:
                list_.append(2)
            if "Petal Width" in selected:
                list_.append(3)
            
            iris = load_iris()

            x = iris.data[:, list_]  # Features (input)
            y = iris.target 

            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio =0.1

            x = torch.Tensor(x)
            y = torch.Tensor(y)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_ratio + val_ratio)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = val_ratio / (val_ratio + test_ratio))
            y_train = y_train.reshape((-1,1))
            y_val = y_val.reshape((-1,1))
            y_test = y_test.reshape((-1,1))

            plt.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis")
            plt.show()
            st.pyplot()

            if st.session_state["train"]:
                iris_model = classification_iris(2, 3)
                loss_fn_iris = CrossEntropyLoss()
                opt = torch.optim.Adam(iris_model.parameters(), lr=0.0001)

                epochs = 10000
                train_loss_model2 = []
                val_loss_model2 = []
                my_bar = st.progress(0, text = txt1)

                for epoch in trange(epochs):
                    iris_model.train()

                    logits = iris_model(x_train)
                    pred = torch.argmax(torch.softmax(logits, dim = 0), axis = 1)

                    loss = loss_fn_iris(logits, y_train.ravel().type(torch.long))
                    train_loss_model2.append(loss.detach().numpy())

                    loss.backward()

                    opt.step()

                    opt.zero_grad()

                    iris_model.eval()
                    
                    if (int(epoch * 100 / epochs) < 80):
                        txt = txt1
                    else:
                        if (int(epoch * 100/ epochs) < 100):
                            txt = txt2
                        else:
                            txt = txt3
                    my_bar.progress(int(epoch * 100/epochs), txt)

                st.markdown("<h2 style ='text-align: center;'>Results</h2>", unsafe_allow_html = True) 
                st.header("Trained Decision Boundary")    
                plot_decision_boundary(iris_model, x,y)
                st.pyplot()
                st.header("Mean Decision Boundary after MC DropOut")
                plot_mean_contour_iris(iris_model, x, iterations, x[:, 0], x[:, 1], y)
                st.pyplot()
                # plot_mean_contour_iris(iris_model, x, iterations)
                # st.pyplot()
                st.header("Standard Deviation after MC DropOut")
                plot_std_contour_iris(iris_model, x, iterations, x[:, 0], x[:, 1], y)
                st.pyplot()
                
                st.session_state["circle"] = False
                st.session_state["blobs"] = False
                st.session_state["generate"] = False
                st.session_state["train"] = False
                st.session_state["iris"] = False
