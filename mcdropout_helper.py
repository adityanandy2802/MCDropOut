import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_vis(model_non_linear, x, y_noisy, iterations, linear = False):
  # iterations = 100
  # plt.scatter(x, y, label = "Actual")
  x_line = torch.tensor(np.linspace(x.min(), x.max())).reshape(-1,1)
  list_2 = []
  for iteration in trange(iterations):
    enable_dropout(model_non_linear)
    predictions = model_non_linear(x_line.type(torch.float32))
    if len(list_2) == 0:
      list_2 = list(predictions.detach().numpy().T)
    else:
      list_2.append(np.array(predictions.squeeze().detach().numpy().T))
    # plt.plot(x_line, model_non_linear(x_line.type(torch.float32)).detach().numpy(), color = "blue", alpha=0.05)
  # plt.legend(loc="best")

  std = []
  for list_ in np.array(list_2).T:
    std.append(list_.std())
  std = np.array(std)

  mean = []
  for list_ in np.array(list_2).T:
    mean.append(list_.mean())
  mean = np.array(mean)

  if not linear:
    plt.plot(x, y_noisy, "--", label = "True Function", color = "black")
  else:
    plt.scatter(x, y_noisy, color = "black", label="True Function", s=10, alpha = 0.2)
  plt.plot(x_line.ravel(), mean, label = "Mean Prediction")
  plt.fill_between(x_line.ravel(), np.array(mean-std), np.array(mean+std), color = "blue", alpha = 0.3, label = "Standard Deviation")
  plt.title("MC Dropout Results")
  plt.legend(loc = "best")
  plt.show()

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
    if x_val is not None and y_val is not None and z_val is not None:
        plt.scatter(x_val, y_val, c=z_val, cmap="viridis", alpha=0.5)
    plt.title("Standard Deviation")
    plt.show()