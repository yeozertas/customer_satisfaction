""" This file includes helper functions """
import os


def get_absolute_folder_path(main_dir, sub_dir_list=None):
    """
    Return absolute path of folder

    Args:
        main_dir (str): Name of main directory
        sub_dir_list (list or None): List of sub directories

    Returns:
        - absolute_path (str) - Path of desired folder
    """

    if sub_dir_list is None:
        sub_dir_list = []

    package_location = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_path = os.path.join(package_location, main_dir)
    for temp_dir in sub_dir_list:
        absolute_path = os.path.join(absolute_path, temp_dir)

    return absolute_path
