'''
/*==========================================================================================
  	Author: 		You-An,Liu
    Email :         xu.6u.30@gmail.com
  	Filename: 		ML_particle_model
	Description:	Global Fitting by Machine Learning
	
	Last update:	2018/AUG/25
//========================================================================================*/
'''

# importing the required modules
import os
import argparse
 
# error messages
INVALID_FILETYPE_MSG = "Error: Invalid file format. %s must be a .txt file."
INVALID_PATH_MSG = "Error: Invalid file path/name. Path %s does not exist."
 
def validate_file(file_name):    
    '''
    validate file name and path.
    '''
    if not valid_path(file_name):
        print(INVALID_PATH_MSG%(file_name))
        quit()
    elif not valid_filetype(file_name):
        print(INVALID_FILETYPE_MSG%(file_name))
        quit()
    return
     
def valid_filetype(file_name):
    # validate file type
    return file_name.endswith('.txt') or file_name.endswith('.csv') or file_name.endswith('.dat')
 
def valid_path(path):
    # validate file path
    return os.path.exists(path) 
 
def show(args):
    # get path to directory
    dir_path = args.show[0]
     
    # validate path
    if not valid_path(dir_path):
        print("Error: No such directory found.")
        exit()
 
    # get text files in directory
    files = [f for f in os.listdir(dir_path) if valid_filetype(f)]
    print("{} text files found.".format(len(files)))
    print('\n'.join(f for f in files))

def main():
    # create parser object
    parser = argparse.ArgumentParser(description = "A text file manager!")
 
    # defining arguments for parser object   
     
    parser.add_argument("-s", "--show", type = str, nargs = 1,
                        metavar = "<path>", default = None,
                        help = "Shows all the data files on specified directory path.\
                        Type '.' for current directory.")

    parser.add_argument("-m", "--mask", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Mask some index (index in parameter.py))")

    parser.add_argument("-d", "--draw", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Draw the distribution of the data")

    parser.add_argument("-g","--with_graph",action='store_true',
                        help = "Some command with this argument will produce graph in ./graph")

    parser.add_argument("-th", "--thinned", type = str, nargs = 2,
                        metavar = ("<file_name>","<partition>"), default = None,
                        help = "Thinned/Smaller the data")

    parser.add_argument("-tr", "--transform", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Show tranform result in Model procedure")

    parser.add_argument("-bt", "--backward_train", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Create Ouput to Input Model and Train Data")

    parser.add_argument("-bp", "--backward_prediction", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Use Ouput to Input Model and Predict Data")

    parser.add_argument("-ft", "--forward_train", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Create Input to Output Model and Train Data")

    parser.add_argument("-fp", "--forward_prediction", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Use Input to Output Model and Predict Data")

    parser.add_argument("-fzt", "--fuzzy_train", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Create Fuzzy Model (Output to Input) and Train Data ( ! Advice Mask Ouput first)")

    parser.add_argument("-fzp", "--fuzzy_prediction", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Use Fuzzy Model (Output to Input) and Predict Data ( ! Advice Mask Ouput first)")

    parser.add_argument("-dvt", "--den_vec_train", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Create Degenerate Vector Model and Train Data ( ! only accept output is 2-dim)")

    parser.add_argument("-dvp", "--den_vec_prediction", type = str, nargs = 1,
                        metavar = ("<file_name>"), default = None,
                        help = "Use Degenerate Vector Model and Predict Data ( ! only accept output is 2-dim)")
    
    '''

        sub argument

    '''
    parser.add_argument("-n", "--name", type = str, nargs = 1,
                        metavar = ("<train_name>"), default = None,
                        help = "important parameter of trian and prediction")

    parser.add_argument("-i","--inputLen", type = str, nargs = 1,
                        metavar = '<Len of input>',
                        help = "important parameter of prediction")

    parser.add_argument("-from","--from_data", type = str, nargs = 1,
                        metavar = '<predict data region>',
                        help = "important parameter of prediction")

    parser.add_argument("-to","--to_data", type = str, nargs = 1,
                        metavar = '<predict data region>',
                        help = "important parameter of prediction")

    parser.add_argument("-w","--weight", type = str, nargs = 1,
                        metavar = '<weight file>',
                        help = "optional parameter of prediction")
    # parse the arguments from standard input
    args = parser.parse_args()
     
    # calling functions depending on type of argument and compute time
    import time
    start_time = time.time()

    if args.show != None:
        show(args)

    elif args.thinned != None:        
        if args.with_graph == True:
            os.system(("python src/data_thinned.py "+args.thinned[0]+" "+args.thinned[1])+" --with_graph")
        else:
            os.system(("python src/data_thinned.py "+args.thinned[0]+" "+args.thinned[1]))

    elif args.transform != None:        
        if args.with_graph == True:
            os.system(("python src/data_transform.py "+args.transform[0]+" --with_graph"))
        else:
            os.system(("python src/data_transform.py "+args.transform[0]))

    elif args.backward_train != None:

        if args.name == None:
            print("command : -bt <file_name> --name <train_name>")
        else:        
            if args.with_graph == True:
                os.system(("python src/backward_model.py "+args.backward_train[0]+" --name "+args.name[0]+" --with_graph"))
            else:
                os.system(("python src/backward_model.py "+args.backward_train[0]+" --name "+args.name[0]))
    elif args.backward_prediction != None:

        if args.name == None or args.inputLen == None or args.from_data == None or args.to_data == None:
            print("command : -bp <file_name> --name <train_name> --inputLen <inputLen> -from <data_from> -to <data_to>")
        else:
            if args.weight == None and args.with_graph == False:
                os.system(("python src/backward_predict.py "+args.backward_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]))
            elif args.weight != None and args.with_graph == False:
                os.system(("python src/backward_predict.py "+args.backward_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]
                +" --weight "+args.weight[0]))
            elif args.weight == None and args.with_graph == True:
                os.system(("python src/backward_predict.py "+args.backward_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]+" --with_graph"))
            elif args.weight != None and args.with_graph == True:
                os.system(("python src/backward_predict.py "+args.backward_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]+" --with_graph"
                +" --weight "+args.weight[0]))
    elif args.forward_train != None:

        if args.name == None:
            print("command : -bt <file_name> --name <train_name>")
        else:        
            if args.with_graph == True:
                os.system(("python src/forward_model.py "+args.forward_train[0]+" --name "+args.name[0]+" --with_graph"))
            else:
                os.system(("python src/forward_model.py "+args.forward_train[0]+" --name "+args.name[0]))
    
    elif args.forward_prediction != None:

        if args.name == None or args.inputLen == None or args.from_data == None or args.to_data == None:
            print("command : -bp <file_name> --name <train_name> --inputLen <inputLen> -from <data_from> -to <data_to>")
        else:
            if args.weight == None and args.with_graph == False:
                os.system(("python src/forward_predict.py "+args.forward_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]))
            elif args.weight != None and args.with_graph == False:
                os.system(("python src/forward_predict.py "+args.forward_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]
                +" --weight "+args.weight[0]))
            elif args.weight == None and args.with_graph == True:
                os.system(("python src/forward_predict.py "+args.forward_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]+" --with_graph"))
            elif args.weight != None and args.with_graph == True:
                os.system(("python src/forward_predict.py "+args.forward_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]+" --with_graph"
                +" --weight "+args.weight[0]))
    elif args.fuzzy_train != None:

        if args.name == None:
            print("command : -bt <file_name> --name <train_name>")
        else:        
            if args.with_graph == True:
                os.system(("python src/fuzzy_model.py "+args.fuzzy_train[0]+" --name "+args.name[0]+" --with_graph"))
            else:
                os.system(("python src/fuzzy_model.py "+args.fuzzy_train[0]+" --name "+args.name[0]))
    
    elif args.fuzzy_prediction != None:

        if args.name == None or args.inputLen == None or args.from_data == None or args.to_data == None:
            print("command : -bp <file_name> --name <train_name> --inputLen <inputLen> -from <data_from> -to <data_to>")
        else:
            if args.weight == None and args.with_graph == False:
                os.system(("python src/fuzzy_predict.py "+args.fuzzy_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]))
            elif args.weight != None and args.with_graph == False:
                os.system(("python src/fuzzy_predict.py "+args.fuzzy_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]
                +" --weight "+args.weight[0]))
            elif args.weight == None and args.with_graph == True:
                os.system(("python src/fuzzy_predict.py "+args.fuzzy_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]+" --with_graph"))
            elif args.weight != None and args.with_graph == True:
                os.system(("python src/fuzzy_predict.py "+args.fuzzy_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]+" --with_graph"
                +" --weight "+args.weight[0]))
    
    elif args.den_vec_train != None:

        if args.name == None:
            print("command : -bt <file_name> --name <train_name>")
        else:        
            if args.with_graph == True:
                os.system(("python src/degenerate_vec_model.py "+args.den_vec_train[0]+" --name "+args.name[0]+" --with_graph"))
            else:
                os.system(("python src/degenerate_vec_model.py "+args.den_vec_train[0]+" --name "+args.name[0]))

    elif args.den_vec_prediction != None:

        if args.name == None or args.inputLen == None or args.from_data == None or args.to_data == None:
            print("command : -bp <file_name> --name <train_name> --inputLen <inputLen> -from <data_from> -to <data_to>")
        else:
            if args.weight == None and args.with_graph == False:
                os.system(("python src/degenerate_vec_predict.py "+args.den_vec_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]))
            elif args.weight != None and args.with_graph == False:
                os.system(("python src/degenerate_vec_predict.py "+args.den_vec_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]
                +" --weight "+args.weight[0]))
            elif args.weight == None and args.with_graph == True:
                os.system(("python src/degenerate_vec_predict.py "+args.den_vec_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]+" --with_graph"))
            elif args.weight != None and args.with_graph == True:
                os.system(("python src/degenerate_vec_predict.py "+args.den_vec_prediction[0]
                +" --name "+args.name[0]+" --inputLen "+args.inputLen[0]
                +" --from "+args.from_data[0]+" --to "+args.to_data[0]+" --with_graph"
                +" --weight "+args.weight[0]))

    elif args.mask != None:
        if args.with_graph == True:
            os.system(("python src/data_mask.py "+args.mask[0]+" --with_graph"))
        else:
            os.system(("python src/data_mask.py "+args.mask[0]))
    
    elif args.draw != None:
        os.system(("python src/draw.py "+args.draw[0]))
    else:
        print("Tips: use \'python main.py -h\' for help!")

    print("use time: --- %s seconds ---" % (time.time() - start_time))
 
if __name__ == "__main__":
    # calling the main function
    main()
