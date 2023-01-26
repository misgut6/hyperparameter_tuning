import gc
import os
import subprocess
import time
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd


def remove_bad_epochs(val_auc, outputs_dir, filename_stem, valid_loss):
    """
    Remove the saved trained model that does not have the lowest valid loss or best performance, to save space.
    :param valid_loss: Validation loss vector
    :param outputs_dir: Directory where model state is saved
    :param filename_stem: The stem of the file name
    Assumption: The file name format will be "{}_epoch{}.pt".format(filename_stem, epoch)
    Where the epoch filename numbers start from 1, not 0
    Requires valid_loss, outputs_dir to be defined in the main script
    """
    # Find the best epoch
    best_auc_epoch = np.argmax(val_auc) + 1
    best_loss_epoch = np.argmin(valid_loss) + 1
    # Show all files in the directory in a list
    output_dir_files = os.listdir(outputs_dir)
    model_filenames_todelete = []
    for filename in output_dir_files:
        if filename.startswith("{}_epoch".format(filename_stem)):
            model_filenames_todelete += [filename]
    # Remove the best epoch filename from the list of files to delete
    model_filenames_todelete.remove("{}_epoch{}.pt".format(filename_stem, best_auc_epoch))
    best_auc_model_filename = "{}/{}_epoch{}.pt".format(outputs_dir, filename_stem, best_auc_epoch)

    if best_auc_epoch != best_loss_epoch:
        model_filenames_todelete.remove("{}_epoch{}.pt".format(filename_stem, best_loss_epoch))
        best_loss_model_filename = "{}/{}_epoch{}.pt".format(outputs_dir, filename_stem, best_loss_epoch)
    else:
        best_loss_model_filename = best_auc_model_filename

    # Delete remaining files that are not from the best epoch
    for filename in model_filenames_todelete:
        subprocess.run(["rm",
                        "{}/{}".format(outputs_dir, filename)], shell=False)
    return best_auc_model_filename, best_loss_model_filename


def classification_metrics(y_true, y_pred):
    # Get AUC

    fpr, tpr, _ = roc_curve(np.vstack(y_true), np.vstack(y_pred))
    roc_auc = auc(fpr, tpr)

    np.save("fpr.npy", fpr)
    np.save("tpr.npy", tpr)

    # Get Accuracy
    accuracy = accuracy_score(np.vstack(y_true), np.round(np.vstack(y_pred)))

    return roc_auc, accuracy


def fit_model_accum(input_model, dataloader, dev):
    input_model.train()
    running_loss = 0
    running_main_loss = 0
    running_reg_loss = 0
    y_true = []  # A list of np arrays containing true labels for each batch
    y_pred = []  # A list of np arrays containing predicted labels for each batch
    last_layer = []
    for i, (X_batch, y_batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X_batch = Variable(X_batch.view(-1, input_model.input_dim)).float().to(dev)
        y_batch = y_batch.float().to(dev)

        input_model.optimizer.zero_grad()

        y_trained, last_layer_preactivation = input_model.forward(X_batch)  # Forward propagation

        # NOTE: last_layer_preactivation is the output layer before activation functions
        # For classification, it is before sigmoid, for regression it is before ReLU or same as output
        # Encode the true label (the default label provided is not encoded)

        # Get the loss and save y_batch (true label) and y_trained (learned label) in list of numpy arrays
        loss, main_loss = input_model.get_loss(y_trained, y_batch)

        # Save the y_true and y_pred
        y_true += list(y_batch.cpu().detach().numpy())
        y_pred += list(y_trained.cpu().detach().numpy())

        # Save the running loss
        running_loss += loss.item()
        running_main_loss += main_loss  # Main loss (BCE or MSE)
        running_reg_loss += loss.item() - main_loss  # Regularization loss

        last_layer += list(last_layer_preactivation.cpu().detach().numpy())

        del X_batch, y_batch

        # Get derivative of the loss dloss/dx with respect to every parameter x which has requires_grad=True
        # The gradients are accumulated into x.grad for each parameter x
        # This basically provides the partial derivative of the loss with respect to each parameter in mode
        # Backpropagation
        loss.backward()
        input_model.optimizer.step()

    loss = running_loss / len(dataloader)  # The average loss for the epoch
    main_loss = running_main_loss / len(dataloader)  # The main loss term
    reg_loss = running_reg_loss / len(dataloader)  # The regularization loss term

    loss_breakdown = {"Main Loss": main_loss, "Reg Loss": reg_loss}

    # Get classification metrics
    roc_auc, accuracy = classification_metrics(y_true, y_pred)

    gc.collect()
    torch.cuda.empty_cache()

    return loss, roc_auc, accuracy, loss_breakdown


# Include the data fields added later for both pre-trained and original models
# The models should have been set up with the correct input dimensions
def validate_model_addfields_copy(trained_model, dataloader, dev, report_results=False):
    trained_model.eval()
    running_loss = 0  # Sum the loss for each different epoch
    running_main_loss = 0
    running_reg_loss = 0
    y_true = []  # A list of np arrays containing true labels for each batch
    y_pred = []  # A list of np arrays containing predicted labels for each batch
    last_layer = []  # A list of np arrays containing the last layer before activation function
    ukb_ids = []  # A list of np arrays containing UKB IDs for the samples
    with torch.no_grad():  # This does not accumulate the gradients for the parameters
        for i, (X_batch, y_batch, X_idx) in tqdm(enumerate(dataloader), total=len(dataloader)):

            X_batch = Variable(X_batch.view(-1, trained_model.input_dim)).float().to(dev)

            y_batch = y_batch.reshape(-1, 1).float().to(dev)

            # The same dimension as y
            y_trained, last_layer_preactivation = trained_model.forward(X_batch)
            # NOTE: last_layer_preactivation is the output layer before activation functions
            # For classification, it is before sigmoid, for regression it is before ReLU or same as output
            # For classification this can be used for percentile vs. prevalence plot

            # Get loss and save y_batch (true label) and y_trained (learned label) in list of numpy arrays
            loss, main_loss = trained_model.get_loss(y_trained, y_batch)

            # Save the y_true and y_pred
            y_true += list(y_batch.cpu().detach().numpy())
            y_pred += list(y_trained.cpu().detach().numpy())

            last_layer += list(last_layer_preactivation.cpu().detach().numpy())

            if report_results:
                # Save the UKB IDs
                ukb_ids += list(X_idx.cpu().detach().numpy())

            running_loss += loss.item()
            running_main_loss += main_loss
            running_reg_loss += loss.item() - main_loss

            del X_batch, y_batch

    validation_loss = running_loss / len(dataloader)  # Total loss
    main_loss = running_main_loss / len(dataloader)  # Main loss (BCE or MSE)
    reg_loss = running_reg_loss / len(dataloader)  # Regularization loss

    # Loss breakdown between main loss (BCE or MSE) and regularization loss
    loss_breakdown = {"Main Loss": main_loss, "Reg Loss": reg_loss}

    # Get classification metrics
    roc_auc, accuracy = classification_metrics(y_true, y_pred)
    # The error is this one: "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')"

    if report_results:  # Provide more data on the validation run
        # Stack the UKB IDs as an array
        ukb_ids = np.array(ukb_ids)
        raw_data = {"y_true": y_true,
                    "y_pred": y_pred,
                    "UKB IDs": ukb_ids}
        return validation_loss, roc_auc, accuracy, last_layer, loss_breakdown, raw_data

    gc.collect()
    torch.cuda.empty_cache()
    # np.save("epoch{}_y_true_val.npy".format(epoch), y_true)

    return validation_loss, roc_auc, accuracy, last_layer, loss_breakdown


def main(model, train_loader, valid_loader, accum_grads, batch_multiples, regression, dev, testing_only,
         datafields_list, encode_multivariate_label, pretrained_model, only_datafields, results_dir=None,
         multitask=False,
         pretrained=False, model_params_dir=None, combination=None):
    # Run the model

    # TODO update if needed
    """
    # Get and save the model weights at layer 1 before training
    for name, params in model.named_parameters():
        if name == "layers_list.0.weight":
            np.save("first_input_weights.npy", params.cpu().detach().numpy())
    """

    # Write a file that the script checks after each epoch which says whether or not to move on to next parameter set
    # TODO set this up
    import time

    start_all_time = time.time()

    tol = 1e-4
    n_iter_no_change = 10  # Number of consecutive epochs change in validation loss <= tol
    n_iter_worse_results = 4  # Number of consecutive epochs change in validation AUC is negative

    train_loss = []
    valid_loss = []
    epoch_count = []
    valid_reg_loss = []  # Regularization loss term
    valid_main_loss = []  # BCE or MSE loss term
    train_reg_loss = []
    train_main_loss = []
    count = 0  # Keep track of changes in loss for each batch of samples across the different epochs
    loss_list = []  # Keep track of loss after each batch of samples across the different epochs
    samples_list = []  # Keep track of the total number of samples
    train_auc = []  # Only if regression is False
    valid_auc = []  # Only if regression is False
    best_val_loss = None  # Initialize this
    no_change_epochs = 0  # This variable counts the consecutive epochs with loss diff < tol
    worse_results_epochs = 0  # This variable counts the consecutive epochs with negative AUC diff
    for epoch in range(model.max_epochs):
        print("\n---------\nEpoch {} of {}".format(epoch + 1, model.max_epochs))
        start_epoch_time = time.time()

        # Fit Model
        if accum_grads:  # Will consider batch_multiples to increase the "batch size"
            print("Not available right now.")
            exit()

        # Run the train function
        train_epoch_loss, train_epoch_auc, train_epoch_accuracy, train_loss_breakdown = fit_model_accum(model,
                                                                                                        train_loader,
                                                                                                        dev)

        print("Train Loss: {}\nTrain AUC: {}\nTrain Accuracy: {}\n".format(train_epoch_loss,
                                                                           train_epoch_auc,
                                                                           train_epoch_accuracy))

        # Validate Model
        # NOTE: Returns last_layer_preactivation for percentile vs. prevalence plot
        # Combined input model
        val_epoch_loss, val_epoch_auc, \
        val_epoch_accuracy, last_layer_preactivation, valid_loss_breakdown = validate_model_addfields_copy(
            model, valid_loader, dev, report_results=False)

        # val_epoch_loss, val_epoch_auc, \
        # val_epoch_accuracy, last_layer_preactivation, valid_loss_breakdown = validate_model(model,
        #                                                                                   valid_loader,
        #                                                                                  dev,
        #                                                                                 report_results=False)

        print("Validation Loss: {}\nValidation AUC: {}\nValidation Accuracy: {}".format(val_epoch_loss,
                                                                                        val_epoch_auc,
                                                                                        val_epoch_accuracy))

        # Check for convergence and early stopping based on loss function
        if epoch >= 1:
            prev_epoch_auc = valid_auc[-1]
            prev_epoch_loss = valid_loss[-1]
            if (val_epoch_loss - prev_epoch_loss) < tol:
                no_change_epochs += 1
                print("If converges, will stop in {} iterations".format(
                    n_iter_no_change - no_change_epochs))
            if (val_epoch_auc - prev_epoch_auc) < 0:
                worse_results_epochs += 1
                print("If performance continues to decrease, will stop in {} iterations".format(
                    n_iter_worse_results - worse_results_epochs
                ))

            else:
                no_change_epochs = 0
                worse_results_epochs = 0

        train_loss.append(train_epoch_loss)
        valid_loss.append(val_epoch_loss)
        epoch_count.append(epoch + 1)

        if not testing_only:
            np.save("{}/epoch_count.npy".format(results_dir), np.array(epoch_count))
            np.save("{}/train_loss.npy".format(results_dir), np.array(train_loss))
            np.save("{}/valid_loss.npy".format(results_dir), np.array(valid_loss))

        # TODO add this

        # Save the breakdown of how regularization vs. MSE / BCE loss change
        train_main_loss.append(train_loss_breakdown["Main Loss"])
        train_reg_loss.append(train_loss_breakdown["Reg Loss"])
        valid_main_loss.append(valid_loss_breakdown["Main Loss"])
        valid_reg_loss.append(valid_loss_breakdown["Reg Loss"])

        if not regression:
            train_auc.append(train_epoch_auc)
            valid_auc.append(val_epoch_auc)

            if not testing_only:
                np.save("{}/train_auc.npy".format(results_dir), train_auc)
                np.save("{}/valid_auc.npy".format(results_dir), valid_auc)

            best_val_auc = np.max(valid_auc)
            auc_epoch = np.argmax(valid_auc)
            print("Best Val AUC: {} at epoch {}".format(best_val_auc, auc_epoch + 1))

        best_val_loss = np.min(valid_loss)
        print("Best Val Loss: {} at epoch {}".format(best_val_loss, np.argmin(valid_loss) + 1))

        if not testing_only:
            np.save("{}/train_main_loss.npy".format(results_dir), train_main_loss)
            np.save("{}/train_reg_loss.npy".format(results_dir), train_reg_loss)
            np.save("{}/valid_main_loss.npy".format(results_dir), valid_main_loss)
            np.save("{}/valid_reg_loss.npy".format(results_dir), valid_reg_loss)

            model_outputfilename = "{}/FeedForwardNN_{}_epoch{}.pt".format(model_params_dir, combination,
                                                                           epoch + 1)
            # torch.save(model.state_dict(), "FeedForwardNN_model.pt")
            torch.save(model.state_dict(), model_outputfilename)
            print("Model State Saved At: {}".format(model_outputfilename))

            # TODO for saving only one model for all combinations, remove the FeedforwardNN_{} from the stem

            # Remove the saved trained model that does not have the lowest loss or best performance, to save space.
            if epoch >= 1:
                best_auc_outputfilename, \
                best_loss_outputfilename = remove_bad_epochs(val_auc=valid_auc, outputs_dir=model_params_dir,
                                                             filename_stem="FeedForwardNN_{}".format(combination),
                                                             valid_loss=valid_loss)

        # ------------------------------------------------------------------------------------
        # How to stop early

        # Stop early if there has been almost no change in loss over many epochs
        if no_change_epochs > n_iter_no_change:
            stopearly = True
        if worse_results_epochs > n_iter_worse_results:
            stopearly = True
        else:
            stopearly = False

        if stopearly:
            best_val_loss = np.min(valid_loss)
            loss_epoch = np.argmin(valid_loss)
            loss_atbestauc_epoch = valid_loss[auc_epoch]
            auc_atbestloss_epoch = valid_auc[loss_epoch]
            if not regression:
                # TODO the experiment 1 trial 1 training had it without auc_epoch+1 or loss_epoch+1
                #  Make it consistent for all of them; either remove before running another trial
                #  or keep the +1 and know that experiment 1 trial 1 is different
                del model
                gc.collect()
                torch.cuda.empty_cache()
                return best_val_auc, auc_epoch + 1, best_val_loss, loss_epoch + 1, epoch + 1, loss_atbestauc_epoch, \
                       auc_atbestloss_epoch
            else:

                return best_val_loss, loss_epoch + 1, epoch + 1

        print("Epoch Time: {}".format(time.time() - start_epoch_time))
        # print("Overall Time: {}".format(time.time() - start_all_time))
        # print("Estimated Time Left {}".format((model.max_epochs - epoch) * (time.time() - start_epoch_time)))

    best_val_loss = np.min(valid_loss)
    loss_epoch = np.argmin(valid_loss)

    if not regression:
        best_val_auc = np.max(valid_auc)
        best_val_loss = np.min(valid_loss)
        auc_epoch = np.argmax(valid_auc)
        loss_atbestauc_epoch = valid_loss[auc_epoch]
        auc_atbestloss_epoch = valid_auc[loss_epoch]
        # TODO the experiment 1 trial 1 training had it without auc_epoch+1 or loss_epoch+1
        #  Make it consistent for all of them
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return best_val_auc, auc_epoch + 1, best_val_loss, loss_epoch + 1, epoch + 1, loss_atbestauc_epoch, auc_atbestloss_epoch

    else:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return best_val_loss, loss_epoch + 1, epoch + 1
