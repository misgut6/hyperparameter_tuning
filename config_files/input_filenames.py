"""
The set input filenames for the prs_analysis_training.py script

"""

# INPUT FILENAMES

project_dir = "/home/monica/projects/prs2_analysis_basic"
phenotype_data_dir = "/datadrive2/UK_Biobank/phenotype_data"

config_filename = "{}/config_files/config_prs2_vars.txt".format(project_dir)

prs2_config_filename = "{}/config_files/prs2_files_config.txt".format(project_dir)

# The disease names should be the same ones in the incident_cases_data files
var_mapping_filename = "{}/config_files/values_mapping.txt".format(project_dir)

# The file with filenames of the disease-specific NG1, PRS1, and labels data
# TODO update later
# disease_data_filename = "{}/config_files/disease_filenames_datadrive.txt".format(project_dir)
disease_data_filename = "{}/config_files/disease_filenames_datadrive.txt".format(project_dir)

# -- General NG2 and PRS2 data files
# Pretrained VAE model for LiteMed
# File location for model state for pretrained LinearVAE model
# Only used for getting the pretrained model
# NOTE: This changes depending on the training of the model
# model_state_filename = "/datadrive/UK_Biobank/phenotype_data/medhistory_pretraining_08.01.22/LinearVAE_epoch489.pt"
model_state_filename = "{}/medhistory_pretraining_08.01.22/LinearVAE_epoch489.pt".format(phenotype_data_dir)
# The location where the model parameters for the LinearVAE model are saved
# model_params_dict_filename = "/datadrive/UK_Biobank/phenotype_data/medhistory_pretraining_08.01.22/current_params_dict.pkl"
model_params_dict_filename = "{}/medhistory_pretraining_08.01.22/current_params_dict.pkl".format(phenotype_data_dir)

# The name of the parent directory for the results of training
# prs_analysis_parentdir = "/localscratch/misgut/phenotype_data/prs_analysis_results"
# prs_analysis_parentdir = "/datadrive/UK_Biobank/phenotype_data/prs2_basic_results"
prs_analysis_parentdir = "/mnt/research1/prs2_basic_results"


# -- General Filenames
# Sample subsets data
# samplesubsets_filename = "/datadrive/UK_Biobank/phenotype_data/pop_sample_subsets_checkzeros_prs.txt"
samplesubsets_filename = "{}/pop_sample_subsets_checkzeros_prs.txt".format(phenotype_data_dir)

# The index file with new data fields
# TODO when moving to the new server change to v5 and move that dataset to the server
# datafields_idx_filename = "/datadrive2/UK_Biobank/phenotype_data/ukb_datafields_withprs_col_idx_v4.txt"
datafields_idx_filename = "{}/ukb_datafields_withprs_col_idx_v5.txt".format(phenotype_data_dir)
# The file with the one-hot-encoded or imputed UK Biobank data fields
# datafields_filename = "/datadrive2/UK_Biobank/phenotype_data/ukb_datafields_withprs_v4.h5"
datafields_filename = "{}/ukb_datafields_withprs_v5.h5".format(phenotype_data_dir)

# The UK Biobank IDs for the medical history file
# medhistory_ids_filename = "/datadrive/UK_Biobank/phenotype_data/medhistory_av1_Merged_ICD10_UKB_IDs.csv"
medhistory_ids_filename = "{}/medhistory_av1_Merged_ICD10_UKB_IDs.csv".format(phenotype_data_dir)
# The medical history data
# medhistory_data = "/datadrive/UK_Biobank/phenotype_data/medhistory_av1_Merged_ICD10_v3.h5"
medhistory_data = "{}/medhistory_av1_Merged_ICD10_v3.h5".format(phenotype_data_dir)
# The fixed pretrained embedding
# fixed_embedding_data = "/datadrive/UK_Biobank/phenotype_data/pretrained_embedding_av1_Merged_ICD10_v3.h5"
fixed_embedding_data = "{}/pretrained_embedding_av1_Merged_ICD10_v3.h5".format(phenotype_data_dir)
