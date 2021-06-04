

MODELS_PATH = r"/home/michael/Cell_Classification/final_models"


DATA_PATH = r"/home/michael/Cell_Classification/Files/Small_Windows_Whitened_23.04/"

DATA_FILE = {'path': r"/home/michael/Cell_Classification/Files/valid + features min-max.xlsx",
             'all_labels': ["head0", "head1", "head2", "mid0", "head3", "combined"],
             'used_labels': ["combined"], 'data_type': "mixed"}


GRAPHS_SAVE_PATH = r"/home/michael/Cell_Classification/final_results"


IMG_SETTINGS = {'img_res': 150, 'img_channels':  3,
                'img_preprocessing':  ['crop']}

BATCH_SIZE = 64
TRAIN_VAL_TEST_SPLIT = (80, 20, 0)
