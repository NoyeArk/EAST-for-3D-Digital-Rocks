import pickle

with open(
    "/home/zql/code/EAST-for-3D-Digital-Rocks/DRP-211/DRSRD Dataset/DRSRD1_3D/DRSRD1_3D/shuffled3D/bin/shuffled3D_train_HR/1543.pt",
    "rb",
) as f:
    data = pickle.load(f)
    print(data.shape)
