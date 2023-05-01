import splitfolders

splitfolders.ratio("../few_shot_dataset_split/val", # The location of dataset
                   output="../few_shot_dataset_split_train", # The output location
                   seed=42, # The number of seed
                   ratio=(.2, .2, .6), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )
