import torch
import numpy as np
import torchvision
from six.moves import urllib # to download MNIST dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score # to calculate results metrics
import pandas as pd
import seaborn as sn

######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.

def download_mnist_dataset():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

######################################################################
# Visualizing the STN results
# ---------------------------
#
# Now, we will inspect the results of our learned visual attention
# mechanism.
#
# We define a small helper function in order to visualize the
# transformations while training.

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def visualize_stn(model, device, test_loader):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

######################################################################
# Live-log training metrics into TensorboardX
# ---------------------------
#   - Training and test losses
#   - Accuracy

def log_to_tensorboard(writer, train_loss, test_loss, acc, epoch):
    # Update loss and accuracy info to tensorboard report for live visualization
    writer.add_scalars('Loss Info', {'train_loss': train_loss,
                                     'test_loss': test_loss}, epoch)
    writer.add_scalar('Test Accuracy', acc, epoch)

######################################################################
# Calculate and visualize training results
# ---------------------------
#   - Accuracy
#   - F1-Score (macro)
#   - Confusion Matrix

def calculate_result_metrics(best_acc, best_acc_targets, best_acc_preds):
    # Compute F1-score
    f1score = f1_score(best_acc_targets, best_acc_preds, average='macro')
    print("\nF1-Score = {:.4f}\n".format(f1score))

    # Compute confusion matrix and save it
    cfm = confusion_matrix(y_true=best_acc_targets, y_pred=best_acc_preds)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    df_cfm = pd.DataFrame(cfm, index=classes, columns=classes)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('Confusion matrix\n(Acc = {:.2f}% | F1-Score = {:.4f}'.format(best_acc, f1score), fontsize=16)
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("./results/cfm.png")