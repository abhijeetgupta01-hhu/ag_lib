# coding:utf-8

__author__ = "Abhijeet Gupta (abhijeet.gupta@gmail.com)"
__copyright__ = "Copyright (C) 2021 Abhijeet Gupta"
__license__ = "GPL 3.0"

###############################################################################################################
#
#  IMPORT STATEMENTS
#
###############################################################################################################

# >>>>>>>>> Native Imports <<<<<<<<<<
import os
import io
import csv
import gzip
import json
import pickle
import zipfile


# >>>>>>>>> Package Imports <<<<<<<<<<


# >>>>>>>>> Local Imports <<<<<<<<<<


###############################################################################################################
#
#  ACTUAL CODE
#
###############################################################################################################
class IO(object):
    """
    Class to read and write data to files

    :read_file() : Function to read a file

    :write_file() : Function to write a file

    :search_files() : Function to recursively search files given a path and an (optional) search string
    """

    __formats = ["txt", "json", "zip", "csv", "gz", "pkl", "pkl.gz"]

    def read_file(self, path=None, filename=None, read_mode="r", format=None, tool=None, encoding="utf-8"):
        """
        Method to read a file, given certain specific parameters

        :path (str) : path from which the file has to be read; can be relative or absolute
        :filename (str) : name of the file; CAN contain the extension
        :format (str) : format in which the file will be read
        :tool (str) : If there are more than 1 packages which can be used to store a file
        :encoding (str) : The encoding in which the file has to be saved
                            The default is "utf-8"
        :can_read (bool): Flag which gives the go-ahead for a file to be read

        :returns : file_data --> return 'Type' is the type returned by the reading package
                    For most files, processed explicitely, the data will be in a list(str)
                    On success, the read data will be returned
                    On Failure, None is be returned
        """
        can_read = True

        if path is None:
            can_read = False
            raise ValueError("The read_file.path attribute cannot accept a NoneType value")
        elif filename is None:
            can_read = False
            raise ValueError("The read_file.filename attribute cannot accept a NoneType value")
        elif format is None:
            can_read = False
            raise ValueError("The read_file.format attribute cannot accept a NoneType value")

        """
        Validating the path
            - if the given path OR the parent path does not seem to be a valid
                directory, an exception will be generated
            - if the path does not exist because the last given directory is a NEW
                directory, then a new directory will be created at the basepath
        """
        if can_read:
            path = os.path.abspath(path)
            if not os.path.isdir(path):
                if not os.path.isdir(os.path.dirname(path)):
                    can_read = False
                    raise ValueError(
                        "The read_file.path attribute does not seem to be a valid directory")
                else:
                    if os.makedirs(path):
                        print("Creating the directory: %s" % os.path.basename(path))

        """
        Validating the file format
        """
        if can_read:
            if format not in self.__formats:
                can_read = False

        """
        Validating the filename and its existence
        """
        if can_read:
            if not filename.endswith(format):
                filename += "." + format
            file = os.path.join(path, filename)
            if not os.path.isfile(file):
                can_read = False
                raise IOError("%s does not exist at the given location" % file)

        """
        Reading the file, if all is good
        """
        read_data = None
        if can_read:
            try:
                if format == "txt":
                    try:
                        with open(file, read_mode) as fp:
                            read_data = [eachLine.strip() for eachLine in fp.readlines() if eachLine]
                    except Exception as e:
                        print("Exception raised in reading text file: %s" % str(e))
                        with open(file, "rb") as fp:
                            read_data = [eachLine.decode(encoding).strip() for eachLine in fp.readlines() if eachLine]

                if format == "csv":
                    with open(file, "r") as fp:
                        read_data = ["\t".join(eachLine).strip() for eachLine in csv.reader(fp, delimiter='\t') if eachLine]

                if format == "json":
                    with open(file, "r") as fp:
                        read_data = json.load(fp)

                if format == "gz":
                    try:
                        with gzip.open(file, "rt", encoding=encoding) as fp:
                            read_data = [eachLine.strip() for eachLine in fp.readlines() if eachLine]
                    except Exception as e:
                        print("Exception raised in reading gz file: %s" % str(e))
                        with gzip.open(file, "rb") as fp:
                            read_data = fp.readlines()

                if format == "pkl":
                    with open(file, "rb") as fp:
                        read_data = pickle.load(fp)

                if format == "pkl.gz":
                    with gzip.open(file, "rb") as fp:
                        read_data = pickle.load(fp)

                if format == "zip":
                    with zipfile.ZipFile(file) as zfp:
                        # Reading from a zip file containing multiple files
                        if len(zfp.namelist()) > 1:
                            for each_file in zfp.namelist():
                                if each_file.endswith("vec") or each_file.endswith("txt"):
                                    # read_data = []
                                    # count = 0
                                    # with zfp.open(each_file, "r") as fp:
                                    #     for each_line in fp:
                                    #         read_data.append(each_line.strip())
                                    #         count += 1
                                    #         if count > 10:
                                    #             break
                                    with zfp.open(each_file, "r") as fp:
                                        read_data = fp.readlines()
                        # Reading from a zip file containing 1 file
                        elif len(zfp.namelist()) == 1:
                            zip_file = zfp.namelist()[0]
                            # read_data = []
                            # count = 0
                            # with zfp.open(zip_file, "r") as fp:
                            #     for each_line in fp:
                            #         read_data.append(each_line.strip())
                            #         count += 1
                            #         if count > 100:
                            #             break
                            with zfp.open(zip_file,"r") as fp:
                                read_data = fp.readlines()
                        else:
                            # Reading directly from the zip file with no compressed files inside
                            try:
                                # read_data = []
                                # count = 0
                                # with zfp.open(each_file, "r") as fp:
                                #     for each_line in fp:
                                #         read_data.append(each_line.strip())
                                #         count += 1
                                #         if count > 10:
                                #             break
                                read_data = zfp.readlines()
                            except Exception as e:
                                can_read = False
                                read_data = None

                    # with io.open(file, 'r', encoding=encoding, newline='\n', errors='ignore') as fp:
                        #return fp.readlines()



            except Exception as e:
                can_read = False
                read_data = None

        if read_data is None:
            print("\tError: Could not read file << %s >>" % file)
        else:
            print("\tRead: << %s >>" % file)
        return read_data



    def write_file(self, path=None, filename=None, write_mode="w", format=None, data=None,
                   tool=None, encoding="utf-8"):
        """
        Method to write a file, given certain specific parameters

        :path (str) : path on which the file has to be written; can be relative or absolute
        :filename (str) : name of the file; CAN contain the extension
        :format (str) : format in which the file will be written
        :tool (str) : If there are more than 1 packages which can be used to store a file
        :encoding (str) : The encoding in which the file has to be saved
                            The default is "utf-8"
        :can_write (bool): Flag which gives the go-ahead for a file to be written

        :returns : True / False (bool)
                    On success, True is returned
                    On Failure, False is returned
        """
        can_write = True

        if path is None:
            can_write = False
            raise ValueError("The write_file.path attribute cannot accept a NoneType value")
        elif filename is None:
            can_write = False
            raise ValueError("The write_file.filename attribute cannot accept a NoneType value")
        elif format is None:
            can_write = False
            raise ValueError("The write_file.format attribute cannot accept a NoneType value")
        elif data is None:
            can_write = False
            raise ValueError("The write_file.data attribute cannot accept a NoneType value")

        """
        Validating the path
            - if the given path OR the parent path does not seem to be a valid
                directory, an exception will be generated
            - if the path does not exist because the last given directory is a NEW
                directory, then a new directory will be created at the basepath
        """
        if can_write:
            path = os.path.abspath(path)
            if not os.path.isdir(path):
                if not os.path.isdir(os.path.dirname(path)):
                    can_write = False
                    raise ValueError(
                        "The write_file.path attribute does not seem to be a valid directory")
                else:
                    if os.makedirs(path):
                        print("Creating the directory: %s" % os.path.basename(path))

        """
        Validating the file format
        """
        if can_write:
            if format not in self.__formats:
                can_write = False

        """
        Validating the filename and its existence
        """
        if can_write:
            if not filename.endswith(format):
                filename += "." + format
            file = os.path.join(path, filename)
            if os.path.isfile(file) and os.path.exists(file):
                print("\t<< %s >> already exists. Overwriting it!" % file)

        """
        Writing the file, if all is good
        """
        if not can_write:
            return can_write
        else:
            try:
                if format == "txt" or format == "csv":
                    try:
                        with open(file, write_mode) as fp:
                            for eachLine in data:
                                fp.write(eachLine.rstrip() + "\n")
                    except Exception as e:
                        print("Exception generated while writing file: %s" % str(e))
                        with open(file, "wb") as fp:
                            for eachLine in data:
                                fp.write((eachLine.rstrip() + "\n").encode(encoding))

                if format == "json":
                    with open(file, "w") as fp:
                        json.dump(data, fp, indent=4)

                if format == "gz":
                    if isinstance(data, list):
                        with gzip.open(file, "wt", encoding=encoding) as fp:
                            for eachLine in data:
                                fp.write(eachLine.strip() + "\n")
                    else:
                        with gzip.open(file, "wb") as fp:
                            fp.write(data)

                if format == "pkl.gz":
                    with gzip.open(file, "wb") as fp:
                        pickle.dump(data, fp)

                if format == "zip":
                    with io.open(file, 'w', encoding=encoding, newline='\n', errors='ignore') as fp:
                        fp.write(data)

                if format == "pkl":
                    with open(file, "wb") as fp:
                        pickle.dump(data, fp)

                print("\tSaved: << %s >>" % file)

            except Exception as e:
                can_write = False
                print("\tError: Could not save file << %s >>.\n\t%s" % (file,str(e)))

            return can_write

    def search_files(self, path=None, search_string=None, match_type=None):
        """
        Module to recursively search for files which match a particular string

        :path (str(abspath)) : A root path on which the files with a particular string will be searched
        :search_string (str) : A partial or exact search string according to which the files have to be searched
        :match_type (str): The kind of search match. Exact string match or Partial string match

        :returns (list(str)): A list of files which match the search criteria
        """
        file_list = []
        if match_type is None or (not match_type == "exact" and not match_type == "partial"):
            raise AttributeError(
                "match_type attribute cannot be a NoneType value. Choices: ['exact', 'partial']")
        else:
            if search_string is None or len(search_string.strip()) < 1:
                search_string = ""
            if match_type == "exact":
                file_list = self.__search_files_exact(path=path, search_string=search_string)
            elif match_type == "partial":
                file_list = self.__search_files_partial(path=path, search_string=search_string)
        return file_list

    def __search_files_exact(self, path=None, search_string=None):
        """
        Module to search for files along a path with a particular name
        EXACT STRING MATCH

        :path (str(abspath)) : A root path on which the files with a particular string will be searched
        :search_name (str) : A string (a file name) which will be searched for in a given directory and added to the filelist

        :returns (list(str)): A list of files which match the search criteria
        """
        file_list = []
        for dirpath, dirs, files in os.walk(path):
            files = [os.path.join(read_path, file) for (read_path, file) in zip([dirpath]*len(files), files) if file.strip() == search_string.strip()]
            for each_file in files:
                if os.path.isfile(each_file):
                    file_list.append(each_file)
        return file_list

    def __search_files_partial(self, path=None, search_string=None):
        """
        Module to search for files along a path with a particular name
        PARTIAL STRING MATCH

        :path (str(abspath)) : A root path on which the files with a particular string will be searched
        :search_name (str) : A string (a file name) which will be searched for in a given directory and added to the filelist

        :returns (list(str)): A list of files which match the search criteria
        """
        file_list = []
        for dirpath, dirs, files in os.walk(path):
            files = [os.path.join(read_path, file) for (read_path, file) in zip([dirpath]*len(files), files) if file.strip().find(search_string) > -1]
            for each_file in files:
                if os.path.isfile(each_file):
                    file_list.append(each_file)
        return file_list

    def make_directory(self, path=None, dir_name=None):
        """
        Module to make a directory at a given path

        :path (str) : The complete path at which the directory will be created
        :dir_name (str) : The name of the new directory

        :returns (bool): True, if the directory was created
                            False, if it wasn't
        """
        is_executed = True

        if path is None:
            is_executed = False
        if dir_name is None:
            is_executed = False

        path = os.path.abspath(path)
        if not os.path.exists(path):
            is_executed = False

        new_dir = os.path.join(path, dir_name)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        return is_executed, new_dir

    def delete_files(self, file_list=None):
        """
        Module to delete a given list of files

        :file_list (list(str)): a list of files to be deleted (path + filename)
            Where, path = abspath

        :returns (bool): True if the files where deleted else False
        """
        is_executed = True

        if not isinstance(file_list, list):
            is_executed = False

        if is_executed:
            not_removed = []
            for each_file in file_list:
                if os.path.exists(each_file):
                    os.remove(each_file)
                else:
                    not_removed.append(each_file)

            # if len(not_removed) > 0:
            #     print("Following files could not be found/deleted:")
            #     print(not_removed)

        return is_executed

    def load_data(path):
        """
        Loads the a .csv file with three columns (or more). And returns a list of for each of the first three columns.

        :path (str): full path to .csv file

        :returns (list, list, list): The three lists of the first three columns
        """
        data = []

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                if line:
                    data.append(line)

        return [x[0] for x in data], [x[1].split()[:-1] for x in data], [x[1][-1] for x in data]


    def load(path, file):
        """
        Loads the a .csv file with three columns (or more). And returns a list of for each of the first three columns.

        :path (str): path to the folder of the file

        :file (str): name of the .csv file

        :returns (list, list, list): The three lists of the first three columns
        """

        fullPath = os.path.join(path,file)

        return load_data(fullPath)
