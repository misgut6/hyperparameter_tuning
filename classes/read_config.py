"""
This script reads the config files and sets up the variables for the script
"""
import os
import pickle
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import loguniform
from classes import dataset_classes, model_classes, train_functions
from functools import reduce


# ----------------
# FUNCTIONS

def get_penalty_weights(group_idx, labels_list, dataset_indices):
    """
    Gets penalty weights dictionary, only viable for binary classification
    It decides the penalty based on the extent of train class imbalance
    :param group_idx: The train set, a numpy array of indices
    :param labels_list: The list of strings with labels. MUST be 1 item in list, and also binary label
    :dataset_indices: The variable returned from setup_indices function
    """
    # Assumption that label must be 1 binary vector (one label)
    label = labels_list[0]
    if len(labels_list) > 1:
        print("There must be no more than 1 item in labels_list")
        exit()

    group_labels = dataset_indices.iloc[group_idx, :][[label]]
    if len(np.unique(group_labels)) > 2:
        print("Must be a binary label")
        exit()

    # Get data frames of cases and controls
    cases = group_labels.loc[group_labels[label] == 1]
    controls = group_labels.loc[group_labels[label] == 0]

    # Count the cases and controls
    case_count = len(cases)
    control_count = len(controls)
    total_count = case_count + control_count

    # Set the penalty based on proportional cases vs controls
    if control_count > case_count:  # Make sure there are more controls
        # For example, if there are 50 controls and 25 cases, the case weight is 2
        case_weight = np.round(control_count / case_count)  # Match case and control ratio
        control_weight = 1
    else:  # If there are more cases than controls, do the opposite. This is unlikely to occur
        control_weight = np.round(case_count / control_count)  # Match case and control ratio
        case_weight = 1

    penalize_dict = {"case_weight": case_weight, "control_weight": control_weight}

    return penalize_dict


def balance_dataset(group_idx, labels_list, dset_indices, undersampling=True):
    """
    Balances the dataset only viable for binary classification
    :param group_idx: The train or validation set, a numpy array of indices
    :param labels_list: The list of strings with labels. MUST be 1 item in list, and also binary label
    :param dataset_indices: The variable returned from setup_indices function
    :undersampling: If True, will remove indices from the largest class
    """
    # Assumption that label must be 1 binary vector (one label)
    label = labels_list[0]
    if len(labels_list) > 1:
        print("There must be no more than 1 item in labels_list")
        exit()

    group_labels = dset_indices.iloc[group_idx, :][[label]]
    # NOTE: These below are only relevant if reset_indices not done after removing NaNs from dataset_indices
    # group_labels = np.array(dataset_indices[label])[group_idx]
    # group_labels = pd.DataFrame(data={"{}".format(label): group_labels})

    if len(np.unique(group_labels)) > 2:
        print("Must be a binary label")
        exit()

    # Get data frames of cases and controls
    cases = group_labels.loc[group_labels[label] == 1]
    controls = group_labels.loc[group_labels[label] == 0]

    # Count the cases and controls
    case_count = len(cases)
    control_count = len(controls)
    print("Cases: {}\nControls: {}".format(case_count, control_count))

    if undersampling:
        # Balance dataset by under sampling
        if control_count > case_count:  # Make sure there are more controls
            controls = controls.iloc[:case_count, :]  # Then match the number of controls to cases
        else:  # If there are more cases than controls, do the opposite. This is unlikely to occur
            cases = cases.iloc[:control_count, :]  # Then match number of cases to controls

    group_indices = list(cases.index) + list(controls.index)

    return np.array(group_indices)


# TODO add the option to save
def get_case_control_count(group_idx, labels_list, dataset_indices, group_name):
    """
        Prints the case and control counts for train or test set
        :param group_idx: The train or validation set, a numpy array of indices
        :param labels_list: The list of strings with labels. MUST be 1 item in list, and also binary label
        :param: dataset_indices: The variable returned from setup_indices function
        :param: group_name: train, validation, test, etc.
        """
    # Assumption that label must be 1 binary vector (one label)
    label = labels_list[0]
    if len(labels_list) > 1:
        print("There must be no more than 1 item in labels_list")
        exit()

    group_labels = dataset_indices.iloc[group_idx, :][[label]]
    # NOTE: These below are only relevant if reset_indices not done after removing NaNs from dataset_indices
    # group_labels = np.array(dataset_indices[label])[group_idx]
    # group_labels = pd.DataFrame(data={"{}".format(label): group_labels})

    if len(np.unique(group_labels)) > 2:
        print("Must be a binary label")
        exit()

    # Get data frames of cases and controls
    cases = group_labels.loc[group_labels[label] == 1]
    controls = group_labels.loc[group_labels[label] == 0]

    # Count the cases and controls
    case_count = len(cases)
    control_count = len(controls)

    print("{} Set\nCases: {}\tControls: {}\n".format(group_name, case_count, control_count))

    return case_count, control_count


def get_sample_subsets(params, group_idx, subsets_file, dataset_indices):
    # TODO if needed, later enable "OR" logic. For now, only "AND" logic is set up
    """
    Assumption: The col names are hard coded in the file
    :params: A dictionary with parameters from subset_params including:
                {"ancestry_subset":
                "ancestry_label":
                "ancestry_code":
                "sex_subset":
                "sex_code":
                "genotype_subset":
                "antibody_subset":
                "removed_subset":
                "av1_nonzero_subset":
                "available_medhistory_subset":
                "incident_nonzero_subset":
                "incident_year":
                }
                :ancestry_subset: A boolean, if True will select a sample subset based on ancestry
                :ancestry_label: "Ancestry_British" (White British), "Ancestry_African", "Ancestry_Indian", and more
                                    based on the data coding 1001 hierarchy
                :ancestry_code: 1 or 0, where 1 means the individual is of the ancestry
                :sex_subset: A boolean, if True will select a sample subset based on sex
                :sex_code: 1 or 0, where 1 means the individual is female
                :genotype_subset: A boolean, if True will select only individuals with available genotype data
                :antibody_subset: A boolean, if True will select only individuals with available antibody data
                :removed_subset: A boolean, if True will select only individuals not removed from UK Biobank
                :av1_nonzero_subset: A boolean, if True will select only individuals with non-zero medical history
                                        at time of assessment visit 1. They might have medical history data later on,
                                        but are still removed because at assessment visit 1 the input is all zeros
                :available_medhistory_subset: A boolean, if True will select only individuals with available medical
                                        history data at any time point. Individuals without medical history data removed
                                        av1_nonzero_subset is a subset of this. Those might have data later on.
                                        This might be useful when using input genotype data. We might not care if we do
                                        not have medical history data for av1, but it matters if there is none at all.
                Important for multivariate regression to predict incident disease risk
                :incident_nonzero_subset: A boolean, if True will select only individuals with non-zero differences
                                        between av1+x (x=incident_year, or number of years after av1), and av1. If
                                        There are no new diagnoses for an individual between these time points we can
                                        exclude the individual when doing regression.
                :incident_year: an integer (otherwise, if incident_year is "NA", incident_nonzero_subset must be False).
                                        The number of years after av1 within which predict disease risk.
                Note: The subsets are not mutually exclusive
    :group_idx: The indices of the samples in the group, that correspond to dataset_indices and classification file
    :subsets_file: The file with population subsets, i.e., "/localscratch/misgut/phenotype_data/pop_sample_subsets.txt"
    """

    # First check which population subsets will be considered

    subset_keys = ["ancestry_subset", "sex_subset", "genotype_subset", "antibody_subset", "removed_subset",
                   "av1_nonzero_subset", "available_medhistory_subset", "incident_nonzero_subset"]

    labels = []
    codes = []
    for key in subset_keys:
        if params[key]:
            if key == "ancestry_subset":
                labels += [params["ancestry_label"]]
                codes += [params["ancestry_code"]]  # Assumption, if ancestry is selected, ancestry_code must be 1 or 0
            elif key == "sex_subset":
                labels += ["Sex_Female"]
                codes += [params["sex_code"]]  # Assumption: if sex is selected, sex_code must be 1 or 0
            elif key == "genotype_subset":
                labels += ["Genotypes_Available"]
                codes += [1]  # Automatically set up to include samples with available genotype data
            elif key == "antibody_subset":
                labels += ["AntibodyData_Available"]
                codes += [1]  # Automatically set up to include samples with available antibody data
            elif key == "removed_subset":
                labels += ["Removed_ID"]
                codes += [0]  # Automatically set up to include non-removed samples
            elif key == "av1_nonzero_subset":
                labels += ["av1_allzero_ids"]
                codes += [0]  # Automatically set up to include samples without all zero vectors at av1
            elif key == "available_medhistory_subset":
                labels += ["unavailable_medhistory_ids"]
                codes += [0]  # Automatically set up to include samples with available medical history data
            elif key == "incident_nonzero_subset":
                if "diff_av1+{}_allzero_ids".format(params["incident_year"]) not in subsets_file.columns:
                    # Check if the incident_year column is available in the dataframe
                    print("The column diff_av1+{}_allzero_ids is not available in the sample_subsets dataframe. "
                          "Change the incident_year input variable".format(params["incident_year"]))
                    exit()
                else:
                    labels += ["diff_av1+{}_allzero_ids".format(params["incident_year"])]
                    codes += [
                        0]  # Automatically set up to include samples with new diagnoses between av1 and incident_year

    i = 0
    subsets_list = []  # A list of data frames with sample subsets
    for item in labels:
        subsets_list += [subsets_file[["UKB ID", item]].iloc[np.where(subsets_file[item] == codes[i])[0], :]]
        i += 1

    # Inner merge the sample subsets data frames and the group_indices (current samples available)
    group_indices = dataset_indices.iloc[group_idx, :]

    if len(subsets_list) > 0:
        subset_df = reduce(lambda left, right: pd.merge(left, right, on=["UKB ID"], how="inner"),
                           subsets_list)
    else:
        return group_idx

    # Get the subset_ids data frame with UKB IDs in the subsets of interest and in the group
    subset_ids = group_indices.iloc[np.where(group_indices["UKB ID"].isin(subset_df["UKB ID"]))[0], :]

    subset_idx = np.array(subset_ids.index)

    return subset_idx


