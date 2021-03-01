import os

train_path = r"G:\aishell\speed_envi_words\normal\quiet_spectrogram\expended_train_6"
validation_path = r"G:\aishell\speed_envi_words\normal\quiet_spectrogram\validation"
test_path = r"G:\aishell\speed_envi_words\normal\quiet_spectrogram\test"

learning_rate = 0  # learning rate
decay = 0     #
momentum = 0

alpha = 0

L1_modulus = 0
L2_modulus = 0
LAMBDA = 0.0

epochs_start = 0   # epoch start
epochs_end = 35     # epoches end
batch_size = 8    # batch size depend on your gpu memary
num_classes = 3   # your dataset classes

image_height = 299
image_wide = 299
num_channels = 3

is_load_pre_model = 0

inception_log_dir = "./inception_logs"
save_model_path = "./model_weight"
log_dir = "./logs"

save_model_history_name = "AiShell_train_history_gelu.csv"
save_model_name = "inception_model_aishell_num_6_loss.h5"
save_model_weight_name = "att_model_aishell_weight_num_classes_3_his_gelu.h5"
save_model_weight_name_cnn = "aishill_six_cnn_weight.h5"
save_model_evaluation = "att_model_aishell_weight_num_classes_3_loss_center_loss1+0.1loss2_earlystopping_evaluation_1.csv"

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)


if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(inception_log_dir):
    os.makedirs(inception_log_dir)

if not os.path.exists(six_cnn_log_dir):
    os.makedirs(six_cnn_log_dir)
