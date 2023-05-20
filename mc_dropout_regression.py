import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from torch.nn import L1Loss, MSELoss
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import trange
import streamlit as st
from mcdropout_helper import enable_dropout, mc_dropout_vis

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h1 style = 'text-align: center;'>MC DropOut</h1>", unsafe_allow_html = True)

torch.manual_seed(42)

class regression_linear(nn.Module):
  def __init__(self, input_features, output_features, hidden_units = 8):
    super().__init__()
    self.relu = nn.ReLU()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        # nn.Dropout(p = 0.1),
        nn.Linear(hidden_units, hidden_units),
        nn.Dropout(p = 0.1),
        nn.Linear(hidden_units, output_features)
    )
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.linear_layer_stack(x)

class regression_non_linear(nn.Module):
  def __init__(self, input_features, output_features, hidden_units = 8):
    super().__init__()
    self.relu = nn.ReLU()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.Sigmoid(),
        nn.Dropout(p = 0.1),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, output_features)
    )
    # self.double
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.linear_layer_stack(x)

# Progress bar status
txt1 = "Model is Training..."
txt2 = "Almost There..."
txt3 = "Done!"

if __name__=="__main__":
    with st.sidebar:
        st.markdown("<h1 style= 'text-align : center;'>Menu</h1>", unsafe_allow_html = True)
        Linear = st.button("Linear", use_container_width = True)
        NonLinear = st.button("Non-Linear", use_container_width = True)
        Generate = st.button("Generate", use_container_width = True, type = "primary")
        Train = st.button("Train", use_container_width = True, type = "primary")

    if "linear" not in st.session_state:
        st.session_state["linear"] = False
    if "nonlinear" not in st.session_state:
        st.session_state["nonlinear"] = False
    if "train" not in st.session_state:
       st.session_state["train"] = False
    if "generate" not in st.session_state:
       st.session_state["generate"] = False

    if Linear:
        st.session_state["linear"] = True
        st.session_state["nonlinear"] = False
    if NonLinear:
        st.session_state["linear"] = False
        st.session_state["nonlinear"] = True
    if st.session_state["linear"] or st.session_state["nonlinear"]:
       if Generate:
          st.session_state["generate"] = True
    if st.session_state["generate"]:
       if Train:
          st.session_state["train"] = True

    if st.session_state["linear"]:
        st.header("Linear Dataset")
        samples = st.slider("Samples", 0, 10000)
        if st.session_state["generate"]:
            x, y = make_regression(n_samples=samples, n_features=1, noise=10)

            # Plot the generated dataset
            plt.scatter(x, y, s=10, color = "black", alpha = 0.2)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Generated Dataset')
            plt.show()
            st.pyplot()

        if st.session_state["train"]:
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

            model_with_dropout = regression_linear(1,1)
            loss_fn = MSELoss()
            opt = torch.optim.Adam(model_with_dropout.parameters(), lr=0.001)

            my_bar = st.progress(0, text = txt1)
            epochs = 10000
            for epoch in trange(epochs):
                model_with_dropout.train()
                pred = model_with_dropout(x_train)
                loss = loss_fn(pred, y_train)
                loss.backward()
                opt.step()
                opt.zero_grad()
                if int((epoch+1)*100/epochs)<80:
                        my_bar.progress(int((epoch+1)*100/epochs), text = txt1)
                else:
                    if int((epoch+1)*100/epochs)<100:
                        my_bar.progress(int((epoch+1)*100/epochs), text = txt2)
                    else:
                        my_bar.progress(int((epoch+1)*100/epochs), text = txt3)
            mc_dropout_vis(model_with_dropout, x, y, 100, True)
            st.pyplot()
            st.session_state["train"] = False
            st.session_state["generate"] = False
            st.session_state["linear"] = False
            st.session_state["nonlinear"] = False

    if st.session_state["nonlinear"]:
        st.header("Non Linear Dataset")
        option = []
        options = ["Sine", "Exponential"]
        option = st.multiselect("Select", options, max_selections = 1)
        
        if option == ["Sine"]:
          k = st.slider("Choose constant", 0, 10)
          n = st.slider("Choose noise", 0.0, 5.0, step = 0.1)
          if st.session_state["generate"]:
              x = np.linspace(0, 2*np.pi, 100)  # X values from 0 to 2*pi
              y = k * np.sin(x)  # Corresponding Y values for the sine curve

              # Add random noise
              noise = np.random.normal(0, n, 100)  # Generate 100 random numbers from a normal distribution with mean 0 and standard deviation 0.1
              y_noisy = y + noise  # Add noise to the original sine curve

              # Plot the sine curve with noise
              plt.plot(x, y_noisy, label='Sine Curve with Noise')
              plt.plot(x, y, linestyle='--', color='r', label='Sine Curve')  # Plot the original sine curve as a reference
              plt.xlabel('X')
              plt.ylabel('Y')
              plt.title('Sine Curve with Noise')
              plt.legend()
              plt.grid(True)
              plt.show()
              st.pyplot()
              
        if option == ["Exponential"]:
            r = st.slider("Choose range", 0, 10)
            k = st.slider("Choose constant", 0, 10)
            n = st.slider("Choose noise", 0.0, 5.0, step = 0.1)
            if st.session_state["generate"]:
                x = np.linspace(0, r, 100)

                # Generate y values with exponential curve and added noise
                noise = np.random.normal(0, n, size=len(x))  # Generate random noise
                y_true = k * np.exp(x)  # True exponential curve without noise
                y_noisy = y_true + noise  # Exponential curve with added noise

                # Plot the true exponential curve and the noisy data
                plt.plot(x, y_true, "--", label='True Exponential Curve')
                plt.plot(x, y_noisy, color='red', label='Noisy Data')

                # Add labels and legend
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()

                # Show the plot
                plt.show()
                st.pyplot()
        if st.session_state["train"]:
            model_non_linear = regression_non_linear(1,1)
            lossfn =  MSELoss()
            optim = torch.optim.Adam(model_non_linear.parameters(), lr=1e-3)
            epochs = 10000

            my_bar = st.progress(0, text = txt1)
            for epoch in trange(epochs):
                model_non_linear.train()
                pred = model_non_linear(torch.tensor(x).reshape(-1,1).type(torch.float))
                loss = lossfn(pred, torch.tensor(y_noisy).type(torch.float32).reshape((-1,1)))
                loss.backward()
                optim.step()
                optim.zero_grad()
                if int((epoch+1)*100/epochs)<80:
                        my_bar.progress(int((epoch+1)*100/epochs), text = txt1)
                else:
                    if int((epoch+1)*100/epochs)<100:
                        my_bar.progress(int((epoch+1)*100/epochs), text = txt2)
                    else:
                        my_bar.progress(int((epoch+1)*100/epochs), text = txt3)
            mc_dropout_vis(model_non_linear, x, y_noisy, 100)
            st.pyplot()
            st.session_state["train"] = False
            st.session_state["generate"] = False
            st.session_state["linear"] = False
            st.session_state["nonlinear"] = False
