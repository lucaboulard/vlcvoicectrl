# -*- coding: utf-8 -*-
"""Generic utility functions.

This module defines utility functions that are used throughout
the project.
"""
import psutil
import os
import os.path
import sys



def print_memory_usage():
    """Prints the RAM usage of the process.
    """
    
    process = psutil.Process(os.getpid())
    print("Memory usage:", _sizeof_fmt(int(process.memory_info().rss)))
    #print(process.memory_info())


def print_sizeof_vars(variables):
    """Prints the size (RAM occupancy) of provided variables.
    
    Reported sizes are not representative for lists and dicts since they only
    store pointers to objects. In this sense this functions work in shallow
    rather than deep mode.
    
    Args:
        variables (list): The list of variable to print. For example, the list
            of local variables can be obtained with locals().
    """
    
    for name, size in sorted(((name, sys.getsizeof(value)) for name,value in variables.items()), key= lambda x: -x[1])[:10]: print("{:>30}: {:>8}".format(name,_sizeof_fmt(size)))


def _sizeof_fmt(num, suffix='B'):
    """Format a number as human readable, based on 1024 multipliers.
    
    Suited to be used to reformat a size expressed in bytes.
    By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254
    
    Args:
        num (int): The number to be formatted.
        suffix (str): the measure unit to append at the end of the formatted
            number.

    Returns:
        str: The formatted number including multiplier and measure unit.
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def list_file_recursive(folder):
    """Gets the names (including path) of files in a folder subtree.
    
    The folder is explored recursively. Subfolder names are not included in the
    list. The file names include paths: can be directly used to open them.
    
    Args:
        folder (str): The folder to search for files.

    Returns:
        list: The list of file names.
    """
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames]