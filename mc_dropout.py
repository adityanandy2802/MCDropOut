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
from mcdropout_helper import enable_dropout, plot_decision_boundary, accuracy_fn, plot_mean_contour, plot_std_contour, plot_mean_contour_iris, plot_std_contour_iris
from mcdropout_helper import custom_calibration_curve, plot_reliability_curve

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
                y_train = y_train.reshape((-1,1))
                y_val = y_val.reshape((-1,1))
                y_test = y_test.reshape((-1,1))

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
                y_train = y_train.reshape((-1,1))
                y_val = y_val.reshape((-1,1))
                y_test = y_test.reshape((-1,1))

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
                plot_mean_contour(model2, x, iterations, x[:, 0], x[:, 1], y)
                st.pyplot()
                # plot_mean_contour(model2, int(x[:,0].min())-1, int(x[:,0].max())+1, int(x[:,1].min())-1, int(x[:,1].max())+1, iterations)
                # st.pyplot()
                st.header("Standard Deviation after MC DropOut")
                plot_std_contour(model2, x, iterations, x[:, 0], x[:, 1], y)
                st.pyplot()
                st.header("Reliability Curve")
                positives, mean_prob = custom_calibration_curve(y, model2(x))
                plot_reliability_curve(mean_prob, positives)
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
