import os
import glob
import configparser
from shutil import rmtree
from process_files import *

if __name__ == '__main__':
    # Reading the configuration parameters from config.ini file.
    config = configparser.ConfigParser()
    config.read(os.path.join("..", "config.ini"))
    in_path = config['projectPreprocessor']['inPath']
    out_path = config['projectPreprocessor']['outPath']
    outputType = config['projectPreprocessor']['outputType']
    maxFileSize = config['projectPreprocessor'].getint('maxFileSize')

    intermediate_path = os.path.join(in_path, "_temp_file_dir_")
    os.mkdir(intermediate_path)

    try:    
        filter_files(in_path, intermediate_path)

        if outputType == "multiple":
            split_files_into_functions_multiple(intermediate_path, out_path, maxFileSize)
        elif outputType == "single":
            split_files_into_functions_single(intermediate_path, out_path, maxFileSize)

    except Exception as e:
        raise Exception(e)

    finally:
        rmtree(intermediate_path)
        for filename in glob.glob("./_temp_*"):
            os.remove(filename) 