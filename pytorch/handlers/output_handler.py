

class FileHandler(object):
    def __init__(self, log_file):
        if log_file != None:
            self.mylogfile = open(log_file, "w")
            self.mylogfile_details = open(log_file + "_best_details.json", "w")

    def myprint(self, message):
        print(message)
        if self.mylogfile != None:
            print(message, file = self.mylogfile)
            self.mylogfile.flush()

    def myprint_details(self, message):
        # print(message)
        if self.mylogfile_details != None:
            print(message, file = self.mylogfile_details)
            self.mylogfile_details.flush()