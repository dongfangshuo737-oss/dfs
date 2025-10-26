import h5py

def print_structure(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print("    Shape:", obj.shape)
        print("    Type:", obj.dtype)

with h5py.File('data/preprocessed/all_uw_data.h5', 'r') as f:
    f.visititems(print_structure)