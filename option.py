class Option:
    def __init__(self):

        # load
        self.folder_path = "./data"
        self.num_person = 1
        self.num_gesture = 12
        self.num_channel = 6
        self.num_row_perpage = 4165

        # slide window
        self.slidewindow_length = 30
        self.slidewindow_stride = 30

        # compare

        self.compare_valid_size=0.2
        self.compare_test_size=0.1
        self.compare_stratify=True
        self.compare_random_state=23
        self.compare_shuffle=True
        self.compare_show_plots=False
        self.compare_bs=64
        self.compare_inplace=True

        # xceptiontime

        self.xceptiontime_valid_size = 0.2
        self.xceptiontime_test_size = 0.1
        self.xceptiontime_stratify = True
        self.xceptiontime_random_state = 23
        self.xceptiontime_shuffle = True
        self.xceptiontime_show_plots = False
        self.xceptiontime_bs = 64
        self.xceptiontime_inplace = True
