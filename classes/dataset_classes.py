# Define the classes
import importlib
import pickle
from classes import model_classes as models
import h5py
import torch
from torch.utils import data
import pandas as pd
import numpy as np
import time
import gc


def setup_indices(classification_file, euro_only=False, labels_list=[]):
    """
    Uses a hard coded classification file. Opens the file, saves columns for labels options, and sets up the
    dataset_indices data frame which contains "800k_Index" for each patient used in the dataset and restricts
    based on ancestry, etc (later can add others).
    It adds labels that can be used for regression or classification
    :param classification_file: The file location of the labels pkl file (can be opened using pd.read_pickle)
    :param euro_only: A boolean True or False
    :param labels_list: The column names of the classification file that indicate what data to extract as labels
    :return:
    """
    # Read the classification dataframe
    # Note that classification_file v2 has fewer samples than v1 because features were merged with it
    # It has limited patients, 488377 (classification v1) vs 487297 (features and classification v2)
    # Otherwise it limits the number of samples
    # TODO change back after debugging
    # classification_file = "/Volumes/Backup/temp_debugging_11.3.2021/labels_df_800k_withidx_v2.pkl"

    # classification_file = "/datadrive/UK_Biobank/genotypes/labels_df_800k_withidx_v2.pkl"
    classification_df = pd.read_pickle(classification_file)

    # Classification options
    class_options = classification_df.columns

    # Set up the dataset_indices dataframe with labels as needed
    if euro_only:
        dataset_indices = classification_df.iloc[np.where(classification_df["European_kmeans"] == 1)[0]][
            ["800k_Index"] + labels_list]
    else:
        dataset_indices = classification_df[["800k_Index"] + labels_list]

    # Delete large file to save memory
    del classification_df
    gc.collect()

    return dataset_indices, class_options


def get_train_test_split(train_pct, valid_pct, dataset_indices):
    """
    This can split the data into 2 or 3 subsets. It is not optimized for cross-validation, and the dataset is not
    balanced since it does not use information about class labels
    Returns a dictionary with information about the split of indices, as well as
    :param train_pct: The percentage of the samples that are in training set
    :param valid_pct: The percentage of the samples that are in validation set
    train_pct + valid_pct = 1, then splits into two; <1, then splits into 3
    :param dataset_indices: A dataframe with columns "800k_Index" and labels
    :return:
    general_params, train_indices, valid_indices, test_indices if train/valid/test
    general_params, train_indices, valid_indices if train/valid
    """
    # Check that percentages add up to at most 1
    if train_pct + valid_pct > 1:
        print("Total percent of train and validation set are greater than 1")
        exit(1)

    total_samples = len(dataset_indices)  # Uses the global variable dataset_indices
    indices = np.arange(total_samples)  # Get range from 0 to total_samples
    np.random.shuffle(indices)  # Randomize the indices
    train_indices = indices[:round(train_pct * len(indices))]
    valid_indices = indices[round(train_pct * len(indices)):round(train_pct * len(indices) + valid_pct * len(indices))]

    if train_pct + valid_pct != 1:  # If it is three groups instead of two
        test_indices = indices[round(train_pct * len(indices) + valid_pct * len(indices)):]

        return train_indices, valid_indices, test_indices

    else:

        return train_indices, valid_indices


# This one is used for visit 1 medical history
class MedHistory_Dataset(data.Dataset):
    """
    This can be used for any HDF5 embedding dataset, from after step 1 or after step 2
    Gives different options
    """

    def __init__(self, indices, filename, ids_filename, return_idx=False, output_labels=None,
                 class_filename=None, datafields_filename=None, datafields_list=None, datafields_idx_filename=None,
                 dev=None, prs2_dict=None):
        """
        :param filename:
        :param output_labels: The labels in the classification file to use, a list of strings
        :param class_filename: A string representing the name of a pkl file. Currently optimized for
                labels_df_800k_withidx_v2.pkl
        :param datafields_filename: A string representing the name of a txt file with two columns. The first column has
                the UK Biobank ID ("UKB ID"), other columns have UK Biobank data field features. Must be selected
                with datafields_list. As in, there needs to be a string here and a non-empty list in datafields_list,
                otherwise ignored. The file should be tab separated
        :param datafields_list: a list of strings, where each string represents the column name for each feature to be
                used by the data loader. Will be checked with datafields_filename. Only those column names also in
                datafields_filename will be added. If it is empty, no additional data field will be added
        :param prs2_dict: A dictionary with details about the PRS2 files

        """

        # NOTE: This is dataset class is based on the assumption that all y data is in file with 800k_Index
        # Requires the labels to be in this labels_df_800k_withidx_v2.pkl

        # ASSUMPTION: The data_file UKB IDs and the class_file UKB IDs are in the same order

        super().__init__()

        # -------------------------------------

        self.data_file = h5py.File(filename, "r")
        hdf_ids = pd.read_csv(ids_filename, sep="\t")[["UKB ID"]]

        self.return_idx = return_idx

        # Get the label for each sample
        self.output_labels = output_labels

        # Has UKB IDs and label as columns
        class_file = pd.read_csv(class_filename, sep="\t")[["UKB ID", self.output_labels[0]]]

        # Remove NaNs from the classification file and reset index
        class_file = class_file.iloc[np.where(~class_file[output_labels[0]].isnull())[0]]
        class_file.reset_index(inplace=True, drop=True)
        class_file = class_file.iloc[indices, :].reset_index(drop=True)

        # Make the UKB IDs of the classification file match those in the data file
        # class_file = pd.merge(class_file, hdf_ids, how="inner", on="UKB ID")

        # class_file.index = class_file["index"]
        class_file = class_file[["UKB ID", self.output_labels[0]]]

        self.len = len(class_file)  # The number of samples is based on the indices

        # self.group_ids = class_file["UKB ID"]

        # Set up the label
        self.output_dim = [len(output_labels)]

        # If additional data fields will be added, include them here
        if datafields_filename is None:
            print("Must enter datafields_filename")
            exit()

        # Load the main data
        print("Loading data...")
        data_file = np.array(self.data_file["main_dataset"])

        # Set it up as a dataframe with the IDS
        data_file = pd.concat([hdf_ids,
                               pd.DataFrame(data=data_file)], axis=1)
        print(class_file)
        # Combine it with the label
        data_file = class_file.merge(data_file, on="UKB ID", how="inner")
        col_names = list(np.arange(0, self.data_file["main_dataset"].shape[1]))

        print("Done loading data")

        # If the datafields file is also going to be used, load it
        # Add the datafields to the analysis
        if len(datafields_list) > 0:

            # Set up a variable for the data file
            datafields_file = h5py.File(datafields_filename, "r")
            # print(datafields_file["main_dataset"])
            # -------
            # Get the columns of the HDF5 file that correspond to the features of interest (of datafields_list)
            # Get the column indices for the features in datafields_list
            datafield_cols_all = pd.read_csv(datafields_idx_filename, sep="\t")

            # Get the list of column indices to use in __getitem__ for the features
            self.col_idx_list = []
            for feature in datafields_list:
                try:
                    self.col_idx_list += [np.where(datafield_cols_all["Col Meaning"] == feature)[0][0]]
                except IndexError:
                    print("Data feature {} is not available".format(feature))  # Data feature name is not available
                    exit()
                    # continue
            idx_df = pd.DataFrame(data={"idx": self.col_idx_list,
                                        "Name": datafields_list}).sort_values(by="idx", ascending=True)
            self.col_idx_list = list(idx_df["idx"])

            datafields_list = list(idx_df["Name"])

            # Checks if we will add these data fields as features
            if len(self.col_idx_list) == 0:
                print("None of the selected features are available in the data field file")
                exit()
            if len(self.col_idx_list) < len(datafields_list):
                print("Some of the selected features are not available in the data fields file")
                exit()

            # -------
            # Combine the UKB IDs with the data fields
            datafields_file = pd.concat([pd.DataFrame(datafields_file["ukb_ids"][:], columns=["UKB ID"]),
                                         pd.DataFrame(datafields_file["main_dataset"][:, self.col_idx_list],
                                                      columns=datafields_list)],
                                        axis=1)
            col_names += datafields_list
            data_file = data_file.merge(datafields_file, how="inner", on="UKB ID")

        # If PRS2 dataset is available
        if prs2_dict is not None:
            all_snps_data = []
            all_snp_ids = []
            for prs in list(prs2_dict.keys()):
                snp_data_filename = prs2_dict[prs]["data"]
                snp_ids_filename = prs2_dict[prs]["snp_ids"]
                samples_filename = prs2_dict[prs]["samples"]  # The same for all PRS keys
                snps_data = np.load(snp_data_filename)
                snp_ids = pd.read_csv(snp_ids_filename, sep="\t")
                all_snps_data += [snps_data]
                all_snp_ids += [snp_ids]
            all_snps_data = np.hstack(all_snps_data)
            all_snp_ids = pd.concat(all_snp_ids, axis=0).reset_index(drop=True)

            # Keep only the unique SNPs
            unique_snp_idx = all_snp_ids.loc[~all_snp_ids.duplicated(subset=["chrpos"], keep="first")].index
            all_snps_data = all_snps_data[:, np.array(unique_snp_idx)]

            unique_snp_ids = all_snp_ids.iloc[unique_snp_idx].reset_index(drop=True)

            prs2_samples = pd.read_csv(samples_filename, sep="\t")
            prs2_samples = prs2_samples.rename(columns={"IID": "UKB ID"})
            prs2_samples = prs2_samples[["UKB ID"]]

            prs2_data = pd.concat([prs2_samples[["UKB ID"]], pd.DataFrame(
                all_snps_data, columns=list(unique_snp_ids["rsid"]))],
                                  axis=1)  # Combine IDs and data
            col_names += list(unique_snp_ids["rsid"])
            # Then make sure to merge the data
            data_file = data_file.merge(prs2_data, how="left",
                                                           on="UKB ID")  # Same order as other data
        else:
            unique_snp_ids = None

        # The features that are not medical history embedding
        if unique_snp_ids is None:
            self.other_features = len(self.col_idx_list)
        else:
            self.other_features = unique_snp_ids.shape[0] + len(self.col_idx_list)

        col_names = pd.DataFrame(data={"Feature": col_names})
        self.col_names = col_names
        self.unique_snp_ids = unique_snp_ids

        self.X_data = torch.from_numpy(np.array(data_file.drop(columns=["UKB ID", self.output_labels[0]]))).to(dev)
        self.y_data = torch.from_numpy(np.array(data_file[self.output_labels[0]]).reshape(-1, 1)).to(dev)
        self.ukb_ids = torch.from_numpy(np.array(data_file["UKB ID"]).reshape(-1, 1)).to(dev)

        # Normalize the dataset
        self.X_data = (self.X_data - torch.min(self.X_data, dim=0)[0]) / (
                torch.max(self.X_data, dim=0)[0] - torch.min(self.X_data, dim=0)[0])

        self.input_dim = self.X_data.shape[1]  # The number of features

        if prs2_dict is not None:
            del prs2_data, all_snps_data, snps_data
        if len(datafields_list) > 0:
            del datafields_file
        del data_file
        gc.collect()

    def __getitem__(self, index):
        # --------------- Retrieve X data for each sample
        # The input is an index from 0 to self.len (total number of indices)
        # NOTE: Only optimized for binary classification
        if self.return_idx:
            return self.X_data[index, :], self.y_data[index], self.ukb_ids[index]

        else:
            return self.X_data[index, :], self.y_data[index]


    def __len__(self):
        return self.len


# This one is used for general data fields
class GeneralDataFields_Dataset(data.Dataset):

    def __init__(self, indices, datafields_filename, datafields_idx_filename, datafields_list,
                 return_idx=False, output_labels=None, class_filename=None,
                 dev=None, prs2_dict=None
                 ):
        """

        :param indices: The numpy array of train_idx, test_idx, etc based on the class_filename
        :param datafields_filename: The string for the location of an HDF5 file containing one dataset labeled ukb_ids
                and another dataset of shape samples by data field, one-hot-encoded and imputed. The dataset is of
                shape (502492, 567), containing only the samples available in all three available phenotype
                datasets (2016, 2019, and 2020)
        :param datafields_idx_filename: The string with the location of a txt file containing the indices and meanings for
                the 567 columns
        :param datafields_list: a list of strings, where each string represents the column name for each feature to be
                used by the data loader. Will be checked with datafields_filename. Only those column names also in
                datafields_filename will be added. If it is empty, no additional data field will be added
        :param return_idx: Whether to return indices as well as X and/or y
        :param output_labels: The labels in the classification file to use, a list of strings
        :param class_filename: A string representing the name of the output classification labels file
        :param dev the device
        :param prs2_dict The dictionary with information about PRS2 files and sample, SNP IDs

        """
        # TODO note that class_file might be modified later to allow for multi-disease prediction

        super().__init__()

        # Read input as self variable
        self.return_idx = return_idx

        # If the necessary files are unavailable, return with an error
        if datafields_filename is None or datafields_idx_filename is None:
            print("Need to input values for datafields_filename, datafields_idx_filename, and datafields_list.")
            exit()

        # Save the HDF5 file as a variable
        self.data_file = h5py.File(datafields_filename, "r")

        # -------
        # Get the columns of the HDF5 file that correspond to the features of interest (of datafields_list)
        # Get the column indices for the features in datafields_list
        datafield_cols_all = pd.read_csv(datafields_idx_filename, sep="\t")

        # Get the list of column indices to use for the features
        self.col_idx_list = []
        datafields_to_use = []
        for feature in datafields_list:
            try:
                self.col_idx_list += [np.where(datafield_cols_all["Col Meaning"] == feature)[0][0]]
                datafields_to_use += [feature]

            except IndexError:
                continue  # Data feature name is not available

        # ------
        # Set up the classification file with the group UKB IDs
        # Remove NaNs from the classification file and reset index
        class_file = pd.read_csv(class_filename, sep="\t")[["UKB ID", output_labels[0]]]
        class_file = class_file.iloc[np.where(~class_file[output_labels[0]].isnull())[0]]
        class_file.reset_index(inplace=True, drop=True)
        class_file = class_file.iloc[indices, :].reset_index()
        class_file.index = class_file["index"]  # Save the index of the original classification file

        if prs2_dict is None and len(self.col_idx_list) == 0:
            print("There are no features to use")
            exit()

        # Check if data fields were identified, anything for NG1 or PRS1
        if len(self.col_idx_list) != 0:

            idx_df = pd.DataFrame(data={"idx": self.col_idx_list,
                                        "Name": datafields_to_use}).sort_values(by="idx", ascending=True)
            self.col_idx_list = list(idx_df["idx"])
            datafields_to_use = list(idx_df["Name"])

            # self.col_idx_list = np.sort(self.col_idx_list)
            # self.input_dim = len(self.col_idx_list)
            # NEED TO SET UP this variable

            # Get the UKB IDs associated with the input data file
            ukb_ids = pd.DataFrame(data=self.data_file["ukb_ids"][:], columns=["UKB ID"])

            # -------
            # Get the row indices of the main data file where the UKB ID is in the classification file
            # To find the row index, first get the UKB ID of the sample and then find the row index using iloc
            ukb_ids = ukb_ids.iloc[np.where(ukb_ids["UKB ID"].isin(class_file["UKB ID"]))[0], :]
            ukb_ids["main_dataset_idx"] = ukb_ids.index
            # The index of this data frame corresponds to the row idx of each sample

            # -------
            # Check that ukb_ids and class_file are the same length (number of samples). If class_file has more samples than
            # ukb_ids after finding the ukb_ids in class_file, the HDF5 dataset samples are not a superset.
            if len(ukb_ids) != len(class_file):
                print("ukb_ids and class_file are not the same length. Make sure the population subsets "
                      "are correctly selected.")
                exit()

            # -------
            # Combine the class file and ukb ids file
            all_current_data = pd.merge(class_file, ukb_ids, how="inner", on="UKB ID")

            self.group_ids = np.array(all_current_data["UKB ID"]).flatten()

            start_time = time.time()
            print("Loading the data...")
            self.X = np.array(self.data_file["main_dataset"])[np.array(all_current_data["main_dataset_idx"]), :]
            self.X = self.X[:, self.col_idx_list]
            self.y = np.array(all_current_data[output_labels])

            self.X = torch.from_numpy(self.X).to(dev)
            self.y = torch.from_numpy(self.y).to(dev)
            self.group_ids = torch.from_numpy(self.group_ids).to(dev)
            print("Done. Time: {}".format(time.time() - start_time))
            del ukb_ids
            gc.collect()

            # -------

            datafields_to_use = pd.DataFrame({"Feature": datafields_to_use})

        # If PRS2 dataset is available
        if prs2_dict is not None:
            all_snps_data = []
            all_snp_ids = []
            for prs in list(prs2_dict.keys()):
                snp_data_filename = prs2_dict[prs]["data"]
                snp_ids_filename = prs2_dict[prs]["snp_ids"]
                samples_filename = prs2_dict[prs]["samples"]  # The same for all PRS keys
                snps_data = np.load(snp_data_filename)
                snp_ids = pd.read_csv(snp_ids_filename, sep="\t")
                all_snps_data += [snps_data]
                all_snp_ids += [snp_ids]
            all_snps_data = np.hstack(all_snps_data)
            all_snp_ids = pd.concat(all_snp_ids, axis=0).reset_index(drop=True)

            # Keep only the unique SNPs
            unique_snp_idx = all_snp_ids.loc[~all_snp_ids.duplicated(subset=["chrpos"], keep="first")].index
            all_snps_data = all_snps_data[:, np.array(unique_snp_idx)]

            unique_snp_ids = all_snp_ids.iloc[unique_snp_idx].reset_index(drop=True)

            prs2_samples = pd.read_csv(samples_filename, sep="\t")
            prs2_samples = prs2_samples.rename(columns={"IID": "UKB ID"})
            prs2_samples = prs2_samples[["UKB ID"]]

            prs2_data = pd.concat([prs2_samples[["UKB ID"]], pd.DataFrame(all_snps_data)], axis=1)  # Combine IDs and data

            # If there are datafields make sure to merge the data
            if len(self.col_idx_list) != 0:
                prs2_data = all_current_data[["UKB ID"]].merge(prs2_data, how="left", on="UKB ID")  # Same order as other data
                prs2_data = torch.from_numpy(np.array(prs2_data.drop(columns=["UKB ID"]))).to(dev)

                # Concatenate the datasets together
                self.X = torch.hstack([self.X, prs2_data])
            else:
                # Use the prs2_samples
                # Combine the class file and prs2_samples file
                all_current_data = pd.merge(class_file, prs2_samples, how="inner", on="UKB ID")
                prs2_data = all_current_data[["UKB ID",
                                              output_labels[0]]].merge(prs2_data, how="left", on="UKB ID")  # Same order as data
                self.X = torch.from_numpy(np.array(prs2_data.drop(columns=["UKB ID", output_labels[0]]))).to(dev)
                self.y = torch.from_numpy(np.array(prs2_data[output_labels[0]]).reshape(-1, 1)).to(dev)
                self.group_ids = torch.from_numpy(np.array(prs2_data["UKB ID"]).flatten()).to(dev)

            # -------

            if len(self.col_idx_list) != 0:
                col_names = pd.concat([unique_snp_ids[["rsid"]].rename(columns={
                    "rsid": "Feature"
                }), datafields_to_use], axis=0).reset_index(drop=True)
            else:
                col_names = unique_snp_ids[["rsid"]].rename(columns={
                    "rsid": "Feature"
                })

        else:
            col_names = datafields_to_use
            unique_snp_ids = None

        # Normalize the data

        self.X = (self.X - torch.min(self.X, dim=0)[0]) / (
                torch.max(self.X, dim=0)[0] - torch.min(self.X, dim=0)[0])

        self.col_names = col_names
        self.unique_snp_ids = unique_snp_ids

        # -------
        # Set up the correct output dim

        self.output_dim = [len(output_labels)]
        self.len = len(class_file)  # The number of samples is based on the indices
        self.input_dim = self.X.shape[1]

        if prs2_dict is not None:
            del prs2_data, all_snps_data, snps_data
            gc.collect()

    def __getitem__(self, index):
        # --------------- Retrieve X data for each sample
        if self.return_idx:
            return self.X[index], self.y[index], self.group_ids[index]
        else:
            return self.X[index], self.y[index]

    def __len__(self):
        return self.len
