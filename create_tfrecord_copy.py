import random
import tensorflow as tf
from dataset_utils import _dataset_exists, write_label_file, _convert_dataset
import os



gpu_id = "2"  
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
#====================================================DEFINE YOUR ARGUMENTS=======================================================================
flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.3, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', None, 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS



def permutate(dataset_dir, threshold=4000, test_size=400):
    print "starting the process"
    dataset_main_folder_list = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,name))]
    dataset_root = os.path.join(dataset_dir, dataset_main_folder_list[0])
    print "#####################################"
    print "dataset_root"
    print dataset_root
    print "#####################################"
    directories = []
    class_names = []
    test_set=[]
    

    for filename in os.listdir(dataset_root):
        path = os.path.join(dataset_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            print "loading subfolders"
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:

        filelist = os.listdir(directory)
        number_of_file = len(filelist)
      
      
        print "number of files"
        print number_of_file
       
        if number_of_file < threshold:
            count=0
            for filename in filelist:
                if count<test_size:
                    path = os.path.join(directory, filename)
                    test_set.append(path)
                    count +=1
                  
                else:
                    path = os.path.join(directory, filename)
                    photo_filenames.append(path)
            
            for i in range(threshold-number_of_file):
                filename = filelist[random.randrange(test_size, number_of_file)]
                path = os.path.join(directory, filename)
                photo_filenames.append(path)
            """    
            print "=================================="            
            print "length of training file"            
            print len(photo_filenames)
            print "length of testing file"            
            print len(test_set)
            print "==================================" 
            """
                
                
        if number_of_file >= threshold:
            for i in range(test_size):
                filename = filelist[i]
                path = os.path.join(directory, filename)
                test_set.append(path)
            
       
            for i in range(test_size,threshold):
                filename = filelist[i]
                path = os.path.join(directory, filename)
                photo_filenames.append(path)
                
            """    
            print "=================================="            
            print "length of training file"            
            print len(photo_filenames)
            print "length of testing file"            
            print len(test_set)
            print "==================================" 
            """
                
    """            
    print "=================================="            
    print "length of training file"            
    print len(photo_filenames)
    print "length of testing file"            
    print len(test_set)
    print "==================================" 
    """
    
    return photo_filenames, sorted(class_names), test_set


def main():

    #==============================================================CHECKS==========================================================================
    #Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    #Check if there is a dataset directory entered
    if not FLAGS.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    #If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir = FLAGS.dataset_dir, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print 'Dataset files already exist. Exiting without re-creating them.'
        return None
    #==============================================================END OF CHECKS===================================================================

    #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names, test_set = permutate(FLAGS.dataset_dir, 10000, 400)

    #Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    #num_validation = int(FLAGS.validation_size * len(photo_filenames))

    #Divide the training datasets into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    random.shuffle(test_set)
    #training_filenames = photo_filenames[num_validation:]
    #validation_filenames = photo_filenames[:num_validation]
    training_filenames=photo_filenames
    validation_filenames=test_set    
    
    
    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, FLAGS.dataset_dir)

    print '\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename)

if __name__ == "__main__":
    main()
