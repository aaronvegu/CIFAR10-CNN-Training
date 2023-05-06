import h5py

with h5py.File('./H5/IWCD_gamma.h5', 'r') as f:
    # variable to control read continuity
    control = 'c'
    while control != 'x':
        # Print the names of all the datasets in the file
        print("Datasets in file:")
        for i, dataset_name in enumerate(f.keys()):
            print(f"{i}. {dataset_name}")
            dataset_len = len(f[dataset_name])
            print(f"   Data count: {dataset_len}")

        # Prompt the user to select a dataset
        selected_dataset_index = int(input("Enter the index of the dataset you want to print: "))

        # Get the selected dataset
        selected_dataset_name = list(f.keys())[selected_dataset_index]
        selected_dataset = f[selected_dataset_name]

        # Print the data in the selected dataset
        print(f"Data in dataset '{selected_dataset_name}':")
        print(selected_dataset[()])

        # ask if continue or stop
        control = input("Want to read another dataset? Enter 'x' to quit, clickENTER to continue: ")

    