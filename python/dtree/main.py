from dtree import *
from id3 import *
import sys


def get_file():
    """
    Tries to extract a filename from the command line.  If none is present, it
    prompts the user for a filename and tries to open the file.  If the file 
    exists, it returns it, otherwise it prints an error message and ends
    execution. 
    """
    # Get the name of the data file and load it into 
    if len(sys.argv) < 2:
        # Ask the user for the name of the file
        print ("Filename: ")
        filename = sys.stdin.readline().strip()
    else:
        filename = sys.argv[1]

    try:
        fin = open(filename, "r")
    except IOError:
        print ("Error: The file '%s' was not found on this system." % filename)
        sys.exit(0)

    return fin

def run_test(fin):
    """
    This function creates a list of exmaples data (used to learn the d-tree)
    and a list of samples (for classification by the d-tree) from the
    designated file.  It then creates the d-tree and uses it to classify the
    samples.  It prints the classification of each record in the samples list
    and returns the d-tree.
    """
    # Create a list of all the lines in the data file
    lines = [line.strip() for line in fin.readlines()]

    # Remove the attributes from the list of lines and create a list of
    # the attributes.
    lines.reverse()
    attributes = [attr.strip() for attr in lines.pop().split(",")]
    target_attr = attributes[-1]
    lines.reverse()

    # Create a list of the data in the data file
    data = []
    for line in lines:
        data.append(dict(zip(attributes,
                             [datum.strip() for datum in line.split(",")])))
        
    # Create the decision tree
    tree = create_decision_tree(data, attributes, target_attr, gain)
    return tree


def run_classify(tree, fin):
    """
    Classify examples data file using decision tree
    """

    # Create a list of all the lines in the data file
    lines = [line.strip() for line in fin.readlines()]
    # Remove the attributes from the list of lines and create a list of
    # the attributes.
    lines.reverse()
    attributes = [attr.strip() for attr in lines.pop().split(",")]
    target_attr = attributes[-1]
    lines.reverse()

    # Create a list of the data in the data file
    examples = []
    for line in lines:
        examples.append(dict(zip(attributes,
                             [datum.strip() for datum in line.split(",")])))
        
    # Classify the records in the examples list
    classification = classify(tree, examples)

    # Print out the classification for each record
    #for item in classification:
    #    print (item)
    return classification

def print_results(data):
    for item in data:
        print (item)    
    

if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    
    print ("Enter Training Data")
    fin = get_file()

    tree = run_test(fin)
    print ("------------------------\n")
    print ("--   Decision Tree    --\n")
    print ("------------------------\n")
    print_tree(tree, "")
    fin.close()    
    print ("\n")
    
    print ("Enter Example Data")
    fin = get_file()
    
    results = run_classify(tree,fin)
    print ("------------------------\n")
    print ("--   Classification   --\n")
    print ("------------------------\n")
    print_results(results)
    fin.close()
    print ("\n")
