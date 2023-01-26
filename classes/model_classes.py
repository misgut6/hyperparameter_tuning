import gc
import pickle

import numpy as np
import torch
from itertools import cycle, islice


class FeedForwardNet(torch.nn.Module):
    """
    Has one hidden layer
    """

    def __init__(self, params):
        """
        :params: Dictionary with model parameters
        :dev: GPU device name string
        NOTE: Class penalization ("penalize_class") in params is optional
        """
        super(FeedForwardNet, self).__init__()

        # The params dict saved as a self variable
        self.params = params
        # Use parameters dictionary to set up the model architecture
        # ASSUMPTION: Each layer is linear, and bias=True (default) which means it learns additive bias term
        self.input_dim = np.int(params["input_dim"])  # The dimension of the input layer
        self.output_dim = params["output_dim"]  # The dimension of the output layer
        hidden_dim = params["hidden_dim"]  # A list of hidden layer dim
        num_hidden_layers = np.int(params["#hidden_layers"])  # A value specifying number of hidden layers

        # Whether to introduce batch normalization to the first layers (not output layer) or not
        self.batch_normalize = params["batch_normalize"]
        self.device = params["device"]  # NOTE: This model is not set up for multi-GPU training for different layers

        # If there are hidden layers
        self.layers_list = []
        self.batch_norms = []
        if num_hidden_layers != 0:
            for num in range(num_hidden_layers):
                if num == 0:
                    # For the first hidden layer, it is taking inputs from the input layer
                    in_layer_features = self.input_dim
                    out_layer_features = hidden_dim[num]  # The current hidden layer
                else:
                    in_layer_features = hidden_dim[num - 1]  # The previous hidden layer
                    out_layer_features = hidden_dim[num]  # The current hidden layer
                self.layers_list.append(torch.nn.Linear(in_layer_features, out_layer_features))
                self.batch_norms.append(torch.nn.BatchNorm1d(in_layer_features, device=self.device))

                # For the last in the list of hidden features, add another with current layer feature
                # count as in_layer_features and middle latent layer as out_layer_features
                if num + 1 == num_hidden_layers:  # If it is the last in the list
                    in_layer_features = hidden_dim[num]
                    out_layer_features = self.output_dim  # Last layer
                    self.layers_list.append(torch.nn.Linear(in_layer_features, out_layer_features))
                    self.batch_norms.append(torch.nn.BatchNorm1d(in_layer_features, device=self.device))
        # If there are no hidden layers, add input to the only layer
        else:
            self.layers_list.append(torch.nn.Linear(self.input_dim, self.output_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.input_dim, device=self.device))

        # Set it as module list
        self.layers_list = torch.nn.ModuleList(self.layers_list)

        # Set up dropout variables
        # Some layers may have dropout and others may not
        self.input_dropout = torch.nn.Dropout(params["input_dropout"])
        self.output_dropout = torch.nn.Dropout(params["output_dropout"])
        self.hidden_dropout_list = torch.nn.ModuleList([torch.nn.Dropout(p) for p in params["hidden_dropout"]])

        # Set up regularization weight and norm values
        # TODO did not add functionality for regularization of input and output layers
        self.input_lambda = params["input_regularize_weight"]  # One value
        self.output_lambda = params["output_regularize_weight"]  # One value
        self.hidden_lambdas = params[
            "hidden_regularize_weight"]  # A list, for model parameters from hidden layers
        self.input_norm = params["input_regularize_norm"]  # One value
        self.output_norm = params["output_regularize_norm"]  # One value
        self.hidden_norm = params["hidden_regularize_norm"]  # A list

        # Use the option for elastic net
        self.elastic_net = params["elastic_net"]
        if self.elastic_net:
            self.elastic_input_lambdas = params["elastic_net_input_lambdas"]
            self.elastic_hidden_lambdas = params["elastic_net_hidden_lambdas"]

        # Set up the Adam optimizer
        # Each parameter, check if it is a string and if so, then make it default, if not then given value
        lr = 1e-3 if type(params["optimizer_lr"]) == str else params["optimizer_lr"]
        betas = (0.9, 0.999) if type(params["optimizer_betas"]) == str else params["optimizer_betas"]
        eps = 1e-8 if type(params["optimizer_eps"]) == str else params["optimizer_eps"]
        weight_decay = 0 if type(params["optimizer_weightdecay"]) == str else params["optimizer_weightdecay"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.batch_size = params["batch_size"]
        self.max_epochs = params["max_epochs"]

        self.mse_loss = params["mse_loss"]

        # Penalizing the classes is an optional parameter
        try:
            self.penalize_class = params["penalize_class"]  # MUST be a boolean
            if type(self.penalize_class) != bool:
                print("penalize_class is optional but if set, must be a boolean True or False")
                exit()
            if self.penalize_class:  # If penalize_class = True, then set weights to penalize
                self.penalize_weights = params["penalize_weights"]  # A dictionary of case_weight and control_weight
        except:
            # If penalize_class key is not in dictionary, set to False
            # Also do this if penalize_class is True but penalize_weights is not in dictionary
            self.penalize_class = False

        if self.mse_loss:
            self.criterion = torch.nn.MSELoss(reduction="none")
        else:
            self.criterion = torch.nn.BCELoss(reduction="none")

        # TODO check the reductions to see what makes sense. Currently set as none, with mean afterwards
        #  may not be the optimal approach, since we are doing: sum(mean of batch x + l1 + l2)x=0:n / n
        """
        # Save the loss as a property of the class of model
        if self.mse_loss:  # MSE for a regression problem
            # TODO look more into this function include non-default reduction or not
            self.criterion = torch.nn.MSELoss(reduction="none")
            self.penalize_class = False  # Set to False
        else:  # BCE for binary classification problem
            # Set criterion to reduction = "none" if penalize_class is True
            if self.penalize_class:
                self.criterion = torch.nn.BCELoss(reduction="none")
            # The default option is "mean"
            else:
                self.criterion = torch.nn.BCELoss(reduction="none")
        """
        if self.penalize_class:
            print("Penalizing the classes")
        else:
            print("Not penalizing the classes")

        # Will run two different outputs, multivariate regression and binary classification
        # The multivariate regression is considered the default output_dim and criterion, and the output dropout is
        # the same for both, as it has not been optimized to be separate for each. But for now it is zero anyways.
        self.multitask = params["multitask"]
        if self.multitask:
            self.bin_output_dim = 1
            if self.penalize_class:
                self.bin_criterion = torch.nn.BCELoss(reduction="none")
            else:
                self.bin_criterion = torch.nn.BCELoss()

        self.return_just_y = False

    def return_just_y_maketrue(self):
        self.return_just_y = True

    def forward(self, X, more_datafields=None):

        # Encoding
        # Assumption: there is a ReLU activation for each layer encoding except for output
        # NOTE: if multitask, returns an additional output
        # TODO decide if there should be an activation for latent layer
        # TODO see what difference it makes to use ReLU as opposed to any other activation function
        # TODO consider adding batch normalization, look into this further
        # TODO maybe add functionality to change activation for layers

        # If additional data fields are included in the forward pass, concatenate it
        # The model should have been already initiated to consider these added features
        if more_datafields is not None:
            X = torch.cat((X, more_datafields), dim=1)

        # NOTE: self.enc_dropout_list is for each layer, and self.enc_list is for transition between layers
        # Have different size overall
        i = 0
        d = 0
        for i in range(len(self.layers_list)):  # The length of the encoding transitions list
            current_layer = self.layers_list[i]
            current_batchnorm = self.batch_norms[i]  # Does not add batch normalization to the output layer

            # Previously added this only after the first layer but resulted in worse performance
            try:  # TODO remove try except clause as needed
                if self.batch_normalize:  # Add if specified, regardless of whether it is one or many layers
                    X = current_batchnorm(X)
            except RuntimeError:
                pass

            if i == 0:  # Include the dropout for the first layer
                if len(self.layers_list) != 1:  # If there are hidden layers

                    X = self.input_dropout(torch.nn.functional.relu(current_layer(X)))  # Add ReLU function to layer

                else:  # If there are no hidden layers (not a deep neural network), the input is the last layer

                    # The output layer before any activation function
                    last_layer_preactivation = current_layer(X).view(-1, self.output_dim)

                    if not self.multitask:
                        # Assumption: ReLU for output layer if regression and sigmoid if classification
                        if self.mse_loss:  # If regression == True
                            # X = self.output_dropout(torch.nn.functional.relu(current_layer(X))).view(-1, self.output_dim)
                            # TODO figure out when to make this ReLU vs no activation
                            X = self.output_dropout(last_layer_preactivation)  # No activation layer
                        else:
                            # The layer has a sigmoid function
                            X = self.output_dropout(torch.sigmoid(last_layer_preactivation))

                        if self.return_just_y:
                            return X
                        else:
                            return X, last_layer_preactivation

                    else:
                        # Use the sigmoid activation for binary classification output and none for multivariate output
                        if self.mse_loss:
                            X_multi = self.output_dropout(last_layer_preactivation)  # No activation layer
                        else:
                            X_multi = self.output_dropout(torch.sigmoid(last_layer_preactivation))

                        X_bin = self.output_dropout(torch.sigmoid(last_layer_preactivation))
                        # X_bin goes into BCE loss and X_multi goes into BCE loss or MSE loss
                        return X_multi, last_layer_preactivation, X_bin

            else:

                dropout = self.hidden_dropout_list[d]  # Dropout for this hidden layer
                if i + 1 != len(self.layers_list):  # If NOT first layer and NOT the last layer before the latent layer
                    X = dropout(torch.nn.functional.relu(current_layer(X)))  # Add dropout and ReLU function to layer

                else:  # If the last layer

                    # The output layer before any activation function
                    last_layer_preactivation = current_layer(X).view(-1, self.output_dim)  # No activation, dropout

                    if not self.multitask:
                        # Assumption: ReLU for output layer if regression and sigmoid if classification
                        if self.mse_loss:  # If regression == True
                            # X = self.output_dropout(torch.nn.functional.relu(current_layer(X))).view(-1, self.output_dim)
                            # TODO figure out when to make this ReLU vs no activation
                            X = self.output_dropout(current_layer(X)).view(-1, self.output_dim)  # No activation layer
                        else:
                            X = self.output_dropout(torch.sigmoid(current_layer(X))).view(-1, self.output_dim)
                    else:
                        # Use the sigmoid activation for binary classification output and none for multivariate output
                        if self.mse_loss:
                            X_multi = self.output_dropout(current_layer(X)).view(-1,
                                                                                 self.output_dim)  # No activation layer
                        else:
                            X_multi = self.output_dropout(torch.sigmoid(current_layer(X))).view(-1, self.output_dim)
                        X_bin = self.output_dropout(torch.sigmoid(current_layer(X))).view(-1, self.bin_output_dim)
                        # X_bin goes into BCE loss and X_multi goes into BCE loss or MSE loss
                        return X_multi, last_layer_preactivation, X_bin

                d += 1
            i += 1

        if self.return_just_y:
            return X
        else:
            return X, last_layer_preactivation

    def get_loss(self, trained_outputs, labels, bin_outputs=None, bin_labels=None, multilabel=False):
        """

        :param trained_outputs: Predicted label
        :param labels: Actual label
        :param bin_outputs: (Optional) The actual binary output, only needed if using multi-task (use trained_outputs for reg.)
        :param bin_labels: (Optional) The actual binary label, only needed if using multi-task (use labels for reg.)
        :param multilabel: (Optional) If the output is multilabel (i.e., 19.1k-dimensional)
        :return:
        """

        # NOTE: by default, if multitask, use multivariate output and label first, and then add the binary ones

        loss = self.criterion(trained_outputs, labels)

        # Added a clause for self.multitask for using self.bin_criterion and self.criterion (for multivariate)
        # So just add them together in the loss function
        if self.multitask:
            loss += self.bin_criterion(bin_outputs, bin_labels)

        if self.penalize_class:  # Setting the weights vector based on penalized weights and saving to device
            # TODO remove boolean
            # multilabel = True
            if not multilabel:
                # NOTE: This is not set up for distributed GPU training by splitting the model
                label_weights = labels.cpu().numpy().flatten()
                label_weights = torch.from_numpy(np.where(
                    label_weights == 1, self.penalize_weights["case_weight"],
                    self.penalize_weights["control_weight"])).reshape(-1, 1).to(self.device)
                # Penalize the loss
                loss = loss * label_weights

            else:
                # For multilabel training, almost all values are zero, so the model can easily learn to classify all
                # 19.1k samples as zeros for all individuals. This needs to be heavily penalized. What is the ratio
                # of 1 to 0 in the input file? Use that to penalize zero NOTE: This is not set up for distributed GPU
                # training by splitting the model
                label_weights = labels.cpu().numpy()
                label_weights = torch.from_numpy(np.where(
                    label_weights == 1, self.penalize_weights["case_weight"],
                    self.penalize_weights["control_weight"])).to(self.device)
                # Penalize the loss
                loss = loss * label_weights

        # Reduce as the mean (Assumption)
        loss = loss.mean()

        # TODO remove after testing
        init_loss = loss.item()

        # Add regularization
        # With regularization, we are penalizing the absolute values of the weights
        # self.layers_list contains variables for each layer transition
        # If the norm is not a string (ex. "None") add regularization term, otherwise continue
        # If the norm is not a string but regularization lambda is zero, will not add to loss
        # Normalization of transition weights
        # Encoding layers regularization

        # NOTE: Elastic net option is not for output layer

        n = 0
        for i in range(len(self.layers_list)):
            if i == 0:  # The first transition between input layer and first hidden layer
                if type(
                        self.input_norm) != str and not self.elastic_net:  # Check that norm is not a string (ex. "None")
                    loss += self.input_lambda * torch.norm(
                        torch.cat([X.view(-1) for X in self.layers_list[i].parameters()]), p=self.input_norm)
                    # print("adding loss first layer", loss, "i", i, "n", n, "lambda", self.input_output_lambda,
                    #     "norm", self.input_output_reg_norm)
                if self.elastic_net:  # If elastic net is selected, ignored the other norm information
                    loss += self.elastic_input_lambdas["L1"] * torch.norm(
                        torch.cat([X.view(-1) for X in self.layers_list[i].parameters()]), p=1) + \
                            self.elastic_input_lambdas["L2"] * torch.norm(
                        torch.cat([X.view(-1) for X in self.layers_list[i].parameters()]), p=2)
            else:
                if i + 1 != len(self.layers_list):  # NOT the last layer
                    if type(self.hidden_norm[n]) != str and not self.elastic_net:
                        # The length of hidden_norms is one less than that of layers_list
                        # It adds Lp normalization to layer transition weights (parameters) and weighted by lambda
                        loss += self.hidden_lambdas[n] * torch.norm(
                            torch.cat([X.view(-1) for X in self.layers_list[i].parameters()]), p=self.hidden_norm[n])
                        # print("adding loss encoding", loss, "i", i, "n", n, "lambda",
                        #     self.enc_lambdas[n], "norm", self.enc_reg_norms[n])
                    if self.elastic_net:  # If elastic net is selected, ignored the other norm information
                        loss += self.elastic_hidden_lambdas[n]["L1"] * torch.norm(
                            torch.cat([X.view(-1) for X in self.layers_list[i].parameters()]), p=1) + \
                                self.elastic_hidden_lambdas[n]["L2"] * torch.norm(
                            torch.cat([X.view(-1) for X in self.layers_list[i].parameters()]), p=2)

                    n += 1
                else:  # It is the output layer
                    if type(self.output_norm) != str:
                        loss += self.output_lambda * torch.norm(
                            torch.cat([X.view(-1) for X in self.layers_list[i].parameters()]), p=self.output_norm)

        # TODO remove the init_loss after testing
        return loss, init_loss



# TODO update this with the one from genotype encoding which allows for one hidden layer
class LinearVAE(torch.nn.Module):
    def __init__(self, params, split_gpus=False):

        """
        The split_gpus functionality allows each layer of the autoencoder to be split into a different GPU, to make it
        easier to train larger models on limited compute power. The script automatically finds how many GPUs are
        made available and what the usage of each one is. The largest layers (input and output) are assigned to
        the GPUs with the most available memory, and the smaller layers are assigned the remaining GPUs.
        It does not work if the inner layers are larger than the outer ones but the Module is not designed fo that.

        """
        super(LinearVAE, self).__init__()

        # Use parameters dictionary to set up the model architecture
        # ASSUMPTION: Each layer is linear, and bias=True (default) which means it learns additive bias term
        self.input_output_dim = np.int32(params["#input_output_features"])  # Number of input and output layer features
        self.latent_dim = params["#latent_layer_features"]
        encoding_dim = params["#encoding_features"]  # A list of number of features for each encoding layer
        decoding_dim = params["#decoding_features"]  # A list of number of features for each decoding layer
        num_encoding_layers = np.int32(params["#encoding_layers"])  # A value specifying number of encoding layers
        num_decoding_layers = np.int32(params["#decoding_layers"])  # A value specifying number of decoding layers

        # If there are encoding layers
        self.enc_list = []
        if num_encoding_layers != 0:
            for num in range(num_encoding_layers):
                if num == 0:
                    # For the first encoding layer, it is taking inputs from the input layer
                    in_layer_features = self.input_output_dim
                    out_layer_features = encoding_dim[num]  # The current encoding layer
                else:
                    in_layer_features = encoding_dim[num - 1]  # The previous encoding layer
                    out_layer_features = encoding_dim[num]  # The current encoding layer
                self.enc_list.append(torch.nn.Linear(in_layer_features, out_layer_features))

                # For the last in the list of encoding features, add another with current layer feature
                # count as in_layer_features and middle latent layer as out_layer_features
                if num + 1 == num_encoding_layers:  # If it is the last in the list
                    in_layer_features = encoding_dim[num]
                    out_layer_features = self.latent_dim * 2  # Multiply it by two to get two variational features
                    self.enc_list.append(torch.nn.Linear(in_layer_features, out_layer_features))

        # If there are no encoding layers, add input to latent layer
        else:
            self.enc_list.append(torch.nn.Linear(in_layer_features, out_layer_features))

        # If there are decoding layers
        if num_decoding_layers != 0:
            self.dec_list = []
            for num in range(num_decoding_layers):
                if num == 0:
                    # For the first decoding layer, it is taking inputs from the latent layer
                    in_layer_features = self.latent_dim
                    out_layer_features = decoding_dim[num]  # The current decoding layer
                else:
                    in_layer_features = decoding_dim[num - 1]  # The previous decoding layer
                    out_layer_features = decoding_dim[num]  # The current decoding layer
                self.dec_list.append(torch.nn.Linear(in_layer_features, out_layer_features))

                # For the last in the list of decoding features, add another with current layer feature
                # count as in_layer_features and output layer as out_layer_features
                if num + 1 == num_decoding_layers:  # If it is the last in the list
                    in_layer_features = decoding_dim[num]
                    out_layer_features = self.input_output_dim
                    self.dec_list.append(torch.nn.Linear(in_layer_features, out_layer_features))
        # If there are no decoding layers
        else:
            self.dec_list.append(torch.nn.Linear(in_layer_features, out_layer_features))

        # ---------------------------------------------
        # Splitting the GPUs
        self.sorted_gpus = self.sort_gpus()  # Get a list of sorted GPU numbers from most to least memory
        # TODO add this as an option
        # Takes a few seconds
        self.split_gpus = split_gpus
        if self.split_gpus:
            print("Setting up distributed GPUs...")
            # Assign the sequential encoding layers to every other GPU starting from the one with most available memory
            # Encoding layers start with the first GPU in list and even indices of the list
            # Do the same with decoding layers but the opposite so the last layer gets the GPU with most memory
            enc_gpus = [item for item in self.sorted_gpus if self.sorted_gpus.index(item) % 2 == 0]
            dec_gpus = [item for item in self.sorted_gpus if self.sorted_gpus.index(item) % 2 != 0]

            self.enc_gpus = list(islice(cycle(enc_gpus), len(self.enc_list)))
            self.dec_gpus = list(islice(cycle(dec_gpus), len(self.dec_list)))[::-1]  # Reverse order

            # Assign a GPU for each enc layer (cut short if needed, or cycle the list)
            for i in range(len(self.enc_list)):
                self.enc_list[i].cuda(self.enc_gpus[i])

            # Assign a GPU for each dec layer (cut short if needed, or cycle the list)
            for i in range(len(self.dec_list)):
                self.dec_list[i].cuda(self.dec_gpus[i])

        # ---------------------------------------------

        # Set them as module lists
        self.enc_list = torch.nn.ModuleList(self.enc_list)
        self.dec_list = torch.nn.ModuleList(self.dec_list)

        # Set up loss function Assumption binary cross entropy
        # torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
        self.criterion = torch.nn.BCELoss(reduction="sum")  # The reduction="sum" is for autoencoders
        # TODO update by going back to BCEWithLogitsLoss later if needed
        # self.criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
        # self.criterion = torch.nn.BCEWithLogitsLoss()

        # Set up dropout variables
        # Some layers may have dropout and others may not
        self.enc_dropout_list = torch.nn.ModuleList([torch.nn.Dropout(p) for p in params["encoding_dropout"]])
        self.dec_dropout_list = torch.nn.ModuleList([torch.nn.Dropout(p) for p in params["decoding_dropout"]])
        self.latent_dropout_list = torch.nn.Dropout(params["latent_dropout"])

        # Set up regularization weight and norm values
        # NOTE: There is no regularization for the input and output layer (Assumption). Should we add this?
        # Assumption: The input and output regularization weights and norms are the same
        self.input_output_lambda = params["input_output_regularize_weight"]  # One value
        self.enc_lambdas = params["encoding_regularize_weight"]  # A list, for model parameters from encoding layers
        self.dec_lambdas = params["decoding_regularize_weight"]  # A list, for model parameters from decoding layers
        self.input_output_reg_norm = params["input_output_regularize_norm"]  # One value
        self.enc_reg_norms = params["encoding_regularize_norm"]  # A list
        self.dec_reg_norms = params["decoding_regularize_norm"]  # A list
        self.reg_params_dict = {"Input and Output Layer Lambdas": self.input_output_lambda,
                                "Encoding Layer Lambdas": self.enc_lambdas,
                                "Decoding Layer Lambdas": self.dec_lambdas,
                                "Input and Output Layer Norms": self.input_output_reg_norm,
                                "Encoding Layer Norms": self.enc_reg_norms,
                                "Decoding Layer Norms": self.dec_reg_norms}

        # Set up the Adam optimizer
        # Each parameter, check if it is a string and if so, then make it default, if not then given value
        lr = 1e-3 if type(params["optimizer_lr"]) == str else params["optimizer_lr"]
        betas = (0.9, 0.999) if type(params["optimizer_betas"]) == str else params["optimizer_betas"]
        eps = 1e-8 if type(params["optimizer_eps"]) == str else params["optimizer_eps"]
        weight_decay = 0 if type(params["optimizer_weightdecay"]) == str else params["optimizer_weightdecay"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.optim_dict = {"Optimizer Type": "Adam", "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        # Batch size and total number of epochs to run
        self.batch_size = params["batch_size"]
        self.max_epochs = params["max_epochs"]

        # TODO if there is an exception such that the batch size causes a memory error because it is too large
        #  add a way to handle it that allows the script to continue running with a smaller batch size
        #  There may be some models that are able to run on larger batch sizes than others due to number of layers, etc
        # TODO set up a way to end the training early if the model converges and set a criterion for convergence

    def sort_gpus(self):
        """
        Find out how many GPUs and order them in order of greatest to least available memory
        Get a list
        Based on 1 - (memory allocated / total memory)
        """
        total_gpus = torch.cuda.device_count()
        gpu_memory = []
        for i in range(0, total_gpus):
            gpu_memory += [(i, 1 - torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory)]
        gpu_memory = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
        sorted_gpus = [item[0] for item in gpu_memory]  # Iterate over the tuples and extract ordered gpus
        return sorted_gpus

    def reparameterize(self, mu, log_var):
        """
        :param mu: The mean from the latent space of the encoder
        :param log_var: The log variance from the latent space of the encoder
        :return: sample
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)  # "randn_like" because we need the same size (epsilon)
        sample = mu + (eps * std)  # Sampling as if it is coming from the input space
        return sample

    def encoding(self, X):
        # Encoding
        # Assumption: there is a ReLU activation for each layer encoding except for latent layer
        # TODO decide if there should be an activation for latent layer
        # TODO see what difference it makes to use ReLU as opposed to any other activation function
        # TODO consider adding batch normalization, look into this further

        # NOTE: self.enc_dropout_list is for each layer, and self.enc_list is for transition between layers
        # Have different size overall
        i = 0
        d = 0
        for i in range(len(self.enc_list)):  # The length of the encoding transitions list
            current_layer = self.enc_list[i]

            if i == 0:  # The assumption is there there is no dropout for the first layer
                if self.split_gpus:  # Assign to first encoding GPU
                    X = torch.nn.functional.relu(current_layer(X.cuda(self.enc_gpus[i])))  # Add ReLU function to layer
                else:
                    X = torch.nn.functional.relu(current_layer(X))  # Add ReLU function to layer

            else:
                dropout = self.enc_dropout_list[d]  # Dropout for this encoding layer
                if i + 1 != len(self.enc_list):  # If NOT first layer and NOT the last layer before the latent layer
                    if self.split_gpus:  # Assign to encoding GPU
                        X = dropout(torch.nn.functional.relu(current_layer(
                            X.cuda(self.enc_gpus[i]))))  # Add dropout and ReLU function to layer
                    else:
                        X = dropout(
                            torch.nn.functional.relu(current_layer(X)))  # Add dropout and ReLU function to layer

                else:  # If the last layer before the latent layer
                    if self.split_gpus:
                        X = dropout(current_layer(X.cuda(self.enc_gpus[i]))).view(-1, 2,
                                                                                  self.latent_dim)  # Assumption: no ReLU for latent layer
                    else:
                        X = dropout(current_layer(X)).view(-1, 2,
                                                           self.latent_dim)  # Assumption: no ReLU for latent layer

                d += 1
            i += 1

        """
        Batch normalization example:
        def __init__(self):
            [...]
            self.linear1 = nn.Linear(in_features=40, out_features=320)
            self.bn1 = nn.BatchNorm1d(num_features=320)
            self.linear2 = nn.Linear(in_features=320, out_features=2)

        def forward(self, input):
            y = F.relu(self.bn1(self.linear1(input)))
            y = F.softmax(self.linear2(y), dim=1)
            return y

        """

        return X

    def decoding(self, Z):
        # Decoding
        X = None  # Empty variable
        i = 0
        d = 0
        for i in range(len(self.dec_list)):
            # There may be dropout in hidden layer
            dropout = self.dec_dropout_list[d] if d < len(self.dec_dropout_list) else None
            current_layer = self.dec_list[i]
            if i == 0:  # If the first decoding layer
                if self.split_gpus:
                    X = dropout(torch.nn.functional.relu(current_layer(Z.cuda(self.dec_gpus[i]))))
                else:
                    X = dropout(torch.nn.functional.relu(current_layer(Z)))

            else:
                if i + 1 != len(self.dec_list):  # If not the last decoding layer
                    if self.split_gpus:
                        X = dropout(torch.nn.functional.relu(current_layer(X.cuda(self.dec_gpus[i]))))
                    else:
                        X = dropout(torch.nn.functional.relu(current_layer(X)))

                elif i + 1 == len(self.dec_list):  # If the last decoding layer
                    # Assumption is that the last activation is sigmoid
                    # No dropout for the output layer
                    # X = torch.sigmoid(current_layer(X))
                    # This option to remove the activation function is fine when using BCEWithLogitsLoss criterion
                    # It is apparently more stable, it includes the sigmoid activation function
                    """
                    if self.split_gpus:
                        X = current_layer(X.cuda(self.dec_gpus[i]))
                    else:
                        X = current_layer(X)
                    """
                    # TODO update by removing the sigmoid later if needed
                    if self.split_gpus:
                        X = torch.sigmoid(current_layer(X.cuda(self.dec_gpus[i])))
                    else:
                        X = torch.sigmoid(current_layer(X))

            d += 1

        return X

    def forward(self, X, get_embedding=False):

        # Run the model with forward propagation
        # Get the middle vector for each sample after encoding
        X = self.encoding(X)

        # Get mu and log_var
        mu = X[:, 0, :]  # The first feature values as mean
        log_var = X[:, 1, :]  # The other feature values as variance

        # Get latent vector through reparameterization
        Z = self.reparameterize(mu, log_var)

        # Get the reconstruction of the input data
        reconstruction = self.decoding(Z)

        if get_embedding:  # Returns the actual embedding of the batch
            return Z
        else:
            return reconstruction, mu, log_var

    def compare_reconst(self, X, reconstruction):
        """
        Calculates for each sample in the batch the distance between X  and reconstruction, and then finds the
        average for the batch
        :param X: The original data (ex. for a given batch)
        :param reconstruction: The reconstruction of the original data
        """
        # TODO make sure this works with split_gpus (may need to assign X and reconstruction)
        # For each sample, compare the distance between the two
        pdist = torch.nn.PairwiseDistance(p=2)
        dist = pdist(X, reconstruction)
        dist_mean = torch.mean(dist)
        dist_std = torch.std(dist)

        return dist_mean, dist_std

    def get_loss(self, X, reconstruction, mu, logvar):
        """
        Adds the reconstruction loss (BCELoss) and the KL-Divergence.
        The KL-Divergence = 0.5*sum(1+log(sigma^2)-mu^2-sigma^2)
        It also adds the regularization terms for all that are not zero.
        :param X: The original data (ex. for a given batch)
        :param reconstruction: The reconstruction of the original data
        :param mu: The mean from the latent vector
        :param logvar: The log variance from the latent vector
        :return: BCE + KLD + [any number of regularization terms]
        """

        # Calculates the sorted GPUs to find the ones with the most available memory, to re-distribute
        if self.split_gpus:
            self.sorted_gpus = self.sort_gpus()

        # TODO look into more of how the Adam optimizer works in comparison to others and how it affects regularization
        if self.split_gpus:
            BCE = self.criterion(reconstruction.cuda(self.sorted_gpus[0]),
                                 X.cuda(self.sorted_gpus[0]))  # Assign to largest GPU
        else:
            BCE = self.criterion(reconstruction, X)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if self.split_gpus:
            loss = BCE.cuda(self.sorted_gpus[0]) + KLD.cuda(self.sorted_gpus[0])
        else:
            loss = BCE + KLD

        # Add regularization
        # With regularization, we are penalizing the absolute values of the weights
        # self.enc_list and self.dec_list contain variables for each layer transition
        # If the norm is not a string (ex. "None") add regularization term, otherwise continue
        # If the norm is not a string but regularization lambda is zero, will not add to loss
        # Normalization of transition weights
        # NOTE: self.enc_list is one longer than self.enc_reg_norms and self.enc_lambdas.
        # The same is for decoding layer variables
        # NOTE: Last decoding layer transition has the same regularization as the first encoding layer transition

        # Encoding layers regularization
        n = 0
        for i in range(len(self.enc_list)):
            if i == 0:  # The first transition between input layer and first hidden layer
                if type(self.input_output_reg_norm) != str:  # Check that norm is not a string (ex. "None")
                    if self.split_gpus:
                        loss += self.input_output_lambda * torch.norm(
                            torch.cat([X.cuda(self.sorted_gpus[0]).view(-1) for X in self.enc_list[i].parameters()]),
                            p=self.input_output_reg_norm)
                    else:
                        loss += self.input_output_lambda * torch.norm(
                            torch.cat([X.view(-1) for X in self.enc_list[i].parameters()]),
                            p=self.input_output_reg_norm)
                        # print("adding loss first layer", loss, "i", i, "n", n, "lambda", self.input_output_lambda,
                        #     "norm", self.input_output_reg_norm)
            else:
                if type(self.enc_reg_norms[n]) != str:  # The length of enc_reg_norms is one less than that of enc_list
                    # It adds Lp normalization to layer transition weights (parameters) and weighted by lambda
                    if self.split_gpus:
                        loss += self.enc_lambdas[n] * torch.norm(
                            torch.cat([X.cuda(self.sorted_gpus[0]).view(-1) for X in self.enc_list[i].parameters()]),
                            p=self.enc_reg_norms[n])

                    else:
                        loss += self.enc_lambdas[n] * torch.norm(
                            torch.cat([X.view(-1) for X in self.enc_list[i].parameters()]), p=self.enc_reg_norms[n])
                        # print("adding loss encoding", loss, "i", i, "n", n, "lambda",
                        #     self.enc_lambdas[n], "norm", self.enc_reg_norms[n])
                n += 1
        # Decoding layers regularization
        n = 0
        for i in range(len(self.dec_list)):
            if i != len(self.dec_lambdas):  # If i is not greater than the largest index in dec_lambas, dec_reg_norms
                if type(self.dec_reg_norms[i]) != str:  # Check that norm is not a string (ex. "None")
                    # It adds Lp normalization to layer transition weights (parameters) and weighted by lambda
                    if self.split_gpus:
                        loss += self.dec_lambdas[i] * torch.norm(
                            torch.cat([X.cuda(self.sorted_gpus[0]).view(-1) for X in self.dec_list[i].parameters()]),
                            p=self.dec_reg_norms[i])

                    else:
                        loss += self.dec_lambdas[i] * torch.norm(
                            torch.cat([X.view(-1) for X in self.dec_list[i].parameters()]),
                            p=self.dec_reg_norms[i])
                        # print("adding loss decoding", loss, "i", i, "n", n,
                        #      "lambda", self.dec_lambdas[i], "norm", self.dec_reg_norms[i])
            else:  # Add the self.input_output_reg_norm and self.input_output_lambda
                if type(self.input_output_reg_norm) != str:
                    if self.split_gpus:
                        loss += self.input_output_lambda * torch.norm(
                            torch.cat([X.cuda(self.sorted_gpus[0]).view(-1) for X in self.dec_list[i].parameters()]),
                            p=self.input_output_reg_norm)
                    else:
                        loss += self.input_output_lambda * torch.norm(
                            torch.cat([X.view(-1) for X in self.dec_list[i].parameters()]),
                            p=self.input_output_reg_norm)
                        # print("adding loss final layer", loss, "i", i, "n", n, "lambda", self.input_output_lambda,
                        #     "norm", self.input_output_reg_norm)

        return loss


class PreTrainedFFNN(torch.nn.Module):
    def __init__(self, pretrained_model, new_ffnn_model, split_gpus=False):
        """
        The split_gpus functionality allows each layer of the autoencoder to be split into a different GPU, to make it
        easier to train larger models on limited compute power. The script automatically finds how many GPUs are
        made available and what the usage of each one is. The largest layers (input and output) are assigned to
        the GPUs with the most available memory, and the smaller layers are assigned the remaining GPUs.
        It does not work if the inner layers are larger than the outer ones but the Module is not designed fo that.

        """
        super(PreTrainedFFNN, self).__init__()

        self.pretrained_model = pretrained_model
        self.ffnn_model = new_ffnn_model

        try:
            del self.pretrained_model.dec_list
            del self.pretrained_model.dec_dropout_list
        except AttributeError:
            pass

        self.input_dim = self.pretrained_model.input_output_dim
        self.output_dim = self.ffnn_model.output_dim
        self.max_epochs = self.ffnn_model.max_epochs  # Uses the FFNN model epochs
        self.optimizer = self.ffnn_model.optimizer  # Uses FFNN model optimizer
        self.more_datafields = None
        self.return_just_y = False

    def get_datafields(self, more_datafields):
        """
        Set up for interpretation
        :param more_datafields:
        :return:
        """
        if self.more_datafields is None:
            self.more_datafields = more_datafields.shape[1]
        else:
            # How many input features
            del self.more_datafields
            gc.collect()
            self.more_datafields = more_datafields.shape[1]

        self.return_just_y = True

    def forward(self, X, more_datafields=None, return_just_y=False):
        """

        :param X: The input data
        :param more_datafields: The per-batch data for the additional features added to the model
        :return:
        """

        # Run the model with forward propagation
        if self.more_datafields is not None:
            X = X[:, :-self.more_datafields]
            more_datafields = X[:, -self.more_datafields:]
            # print(more_datafields)
            # print(more_datafields.shape)
            # print(np.any(more_datafields.cpu().detach().numpy()))

        # Set up a way for the pretrained model to only go through to get inner hidden layer
        X = self.pretrained_model.encoding(X)  # Forward propagation through pretrained model

        # Get mu and log_var
        mu = X[:, 0, :]  # The first feature values as mean
        log_var = X[:, 1, :]  # The other feature values as variance

        # Get latent vector through reparameterization
        Z = self.pretrained_model.reparameterize(mu, log_var)

        # TODO update this to fix it
        # If additional data fields are included in the forward pass, concatenate it
        # The model should have been already initiated to consider these added features
        # TODO change to this
        # if more_datafields is not None:
        if self.more_datafields is not None:
            Z = torch.cat((Z, more_datafields), dim=1)

        # Forward propagation through new model
        y, last_layer_presigmoid = self.ffnn_model.forward(Z)

        # if return_just_y:
        if self.return_just_y:
            return y
        else:
            return y, last_layer_presigmoid

    def get_loss(self, y_pred, y_actual):
        # Get the FFNN model loss over just the new model features
        # TODO note this does not account for fine tune, regularization loss is based only on ffnn model
        #  could add specific regularization term for each layer if the autoencoder will be updated along with
        #  the feedforward network
        #  It is set up to just take the current hidden features and use them in the ffnn model
        loss = self.ffnn_model.get_loss(y_pred, y_actual)

        return loss


# Designed for using encoder or decoder functionality based on pretrained LinearVAE
# TODO update this to enable use of the encoder (19.1k to 500) or decoder (500 to 19.1k) as needed
class PreTrainedVAE(torch.nn.Module):
    def __init__(self, vae_params_filename, vae_state_filename, device):
        """
        The split_gpus functionality allows each layer of the autoencoder to be split into a different GPU, to make it
        easier to train larger models on limited compute power. The script automatically finds how many GPUs are
        made available and what the usage of each one is. The largest layers (input and output) are assigned to
        the GPUs with the most available memory, and the smaller layers are assigned the remaining GPUs.
        It does not work if the inner layers are larger than the outer ones but the Module is not designed fo that.

        """
        super(PreTrainedVAE, self).__init__()

        # Get the model_params_dict for the pretrained model
        with open(vae_params_filename, "rb") as file:
            vae_model_params_dict = pickle.load(file)

        # Initialize the pretrained model
        # Input the dict into the model
        self.pretrained_model = LinearVAE(vae_model_params_dict).to(device)

        # Load the model state dict
        self.pretrained_model.load_state_dict(
            # torch.load(vae_state_filename, map_location=dev))  # Specify GPU to load to
            torch.load(vae_state_filename, map_location=device))

        # Freeze the model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # del self.pretrained_model.dec_list
        # del self.pretrained_model.dec_dropout_list

        self.input_dim = self.pretrained_model.input_output_dim
        self.latent_dim = np.int64(self.pretrained_model.enc_list[-1].weight.shape[0] / 2)
        self.output_dim = self.input_dim

    def decode(self, Z):
        # TODO consider updating the decoding activation function?
        # Get the reconstruction of the input data
        reconstruction = self.pretrained_model.decoding(Z)

        return reconstruction

    def encode(self, X):
        """

        :param X: The input data
        :param more_datafields: The per-batch data for the additional features added to the model
        :return:
        """
        # Run the model with forward propagation

        # Set up a way for the pretrained model to only go through to get inner hidden layer
        X = self.pretrained_model.encoding(X)  # Forward propagation through pretrained model

        # Get mu and log_var
        mu = X[:, 0, :]  # The first feature values as mean
        log_var = X[:, 1, :]  # The other feature values as variance

        # Get latent vector through reparameterization
        Z = self.pretrained_model.reparameterize(mu, log_var)

        return Z

    def get_loss(self, y_pred, y_actual):
        # Get the FFNN model loss over just the new model features
        # TODO note this does not account for fine tune, regularization loss is based only on ffnn model
        #  could add specific regularization term for each layer if the autoencoder will be updated along with
        #  the feedforward network
        #  It is set up to just take the current hidden features and use them in the ffnn model
        loss = self.ffnn_model.get_loss(y_pred, y_actual)

        return loss
