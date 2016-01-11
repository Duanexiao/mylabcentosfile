# -*- coding: utf-8 -*-
import os


def load_mat_file(matfile):
    from scipy.io import loadmat,  whosmat
    return_variable_dict = {}
    variables_in_matfile = whosmat(matfile)  #获取mat文件中的所有变量,返回的是一个列表，每个列表中是一个元组
    file_from_mat = loadmat(matfile)
    for variable_i in variables_in_matfile:
        variable_i_value = file_from_mat[variable_i[0]]
        return_variable_dict[variable_i[0]] = variable_i_value 
        globals()[variable_i[0]] = variable_i_value  #此处相当于将该变量声明并赋值后将其设置为全局变量
        #locals()[variable_i[0]] = variable_i_value  #此处只是相当于将该变量声明并赋值，并且locals是只读的，并不能修改该变量的原始值！！！！
                                                     #只需在前面声明赋值了此变量，并在此行代码后就会发现这个
    return return_variable_dict

path = 'C:\\Users\\duane\\Documents\\MATLAB'         #mat文件的路径
os.chdir(path)
all_files = os.listdir(path)
all_mat_files = [i for i in all_files if i.endswith('.mat')]
for mat_file_i in all_mat_files:
    variable_dict_i = load_mat_file(mat_file_i)
    #for key_i in variable_dict_i.keys():
        #locals()[key_i] = variable_dict_i[key_i]
#for i in locals().items():
#    print i                #打印所有的局部变量
print a,'\n\n',b,'\n\n',c   #mat文件中的所有变量


