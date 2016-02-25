from pybrain.structure import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import os
from scipy.io import loadmat,  whosmat




def fnn_regression(fnn_net_structure):
    if len(fnn_net_structure[1]) == 3 and isinstance(fnn_net_structure[1],list):
        hidden_layer_names = ['hiddenLayer'+str(i) for i in range(len(fnn_net_structure[1]))]
        hidden_connection_name = ['h'+str(i)+'_to_'+'h'+str(i+1) for i in range(len(hidden_layer_names)-1)]
    else:
        raise 'hidden layer units number must save as list and this function do not supports without hidden layer mode'
    fnn = FeedForwardNetwork()
    inLayer = LinearLayer(fnn_net_structure[0], name='inLayer')  
    for hidden_layer_sequence, units_num in enumerate(fnn_net_structure[1]):
        globals()[hidden_layer_names[hidden_layer_sequence]] = SigmoidLayer(units_num, name=hidden_layer_names[hidden_layer_sequence])
        print hidden_layer_names[hidden_layer_sequence]
    outLayer = LinearLayer(fnn_net_structure[-1], name='outLayer')
    
    fnn.addInputModule(inLayer)
    for layer_name in hidden_layer_names:
        fnn.addModule(globals()[layer_name])
    fnn.addOutputModule(outLayer)
    
    in_to_hidden = FullConnection(inLayer, hiddenLayer0)
    for i in range(len(hidden_layer_names)-1):
        globals()[hidden_connection_name[i]] = FullConnection(globals()[hidden_layer_names[i]], globals()[hidden_layer_names[i+1]])
    hidden_to_out = FullConnection(globals()[hidden_layer_names[-1]], outLayer)
    
    fnn.addConnection(in_to_hidden)
    for names in hidden_connection_name:
        fnn.addConnection(globals()[names])
    fnn.addConnection(hidden_to_out)
    fnn.sortModules()
    return fnn


def fnn_datasets(data_x, label_y,train_test_rate):
    input_demension = np.shape(data_x)[1]
    target_demension = np.shape(label_y)[1]
    print input_demension, target_demension
    DS = SupervisedDataSet(input_demension, target_demension)    #定义数据集的格式是三维输入，一维输出
    for i in range(np.shape(data_x)[0]):
        DS.addSample(data_x[i], label_y[i])
    dataTrain, dataTest = DS.splitWithProportion(train_test_rate)
    #xTrain, yTrain = dataTrain['input'], dataTrain['target']
    #xTest, yTest = dataTest['input'], dataTest['target']
    return dataTrain, dataTest


def fnn_train(fnn, dataTrain,max_iter_num,lr):
    trainer = BackpropTrainer(fnn, dataTrain, verbose=True, learningrate=lr)   #verbose = True即训练时会把Total error打印出来,库里默认训练集和验证集的比例为4:1,可以在括号里更改
    trainer.trainUntilConvergence(maxEpochs=max_iter_num)
    return fnn




def fnn_predict(fnn,test_dataset):
    predict_value = []
    true_value = []
    for i in range(np.shape(test_dataset['input'])[0]):
        predict_value.append(fnn.activate(test_dataset['input'][i]))
        true_value.append(test_dataset['target'][i])
    return predict_value, true_value
    


def regression_evaluation(predict_value, true_value):
    from sklearn import metrics
    #平均绝对误差(MAE)
    print "MAE:", metrics.mean_absolute_error(true_value, predict_value)
    #均方误差(MSE)
    print "MSE:", metrics.mean_squared_error(true_value, predict_value)
    #均方根误差(RMSE)
    print "RMSE:", np.sqrt(metrics.mean_squared_error(true_value, predict_value))

 

def print_parameters(fnn):
    for mod in fnn.modules:
        print "Module:", mod.name
        if mod.paramdim > 0:
            print "--parameters:", mod.params
        for conn in fnn.connections[mod]:
            print "-connection to", conn.outmod.name
            if conn.paramdim > 0:
                print "- parameters", conn.params
        if hasattr(fnn, "recurrentConns"):
            print "Recurrent connections"
            for conn in fnn.recurrentConns:             
                print "-", conn.inmod.name, " to", conn.outmod.name
                if conn.paramdim > 0:
                    print "- parameters", conn.params


def load_mat_file(mat_file_path):
    all_files = os.listdir(mat_file_path)
    for i in all_files:
        if i.endswith('.mat'):
            data_file = i
    return_variable_dict = {}
    variables_in_matfile = whosmat(data_file)    
    file_from_mat = loadmat(data_file)
    for variable_i in variables_in_matfile:
        variable_i_value = file_from_mat[variable_i[0]]
        return_variable_dict[variable_i[0]] = variable_i_value
    data_x = return_variable_dict['data_x']
    label_y = return_variable_dict['label_y']
    return data_x, label_y
  





fnn_net_structure = [8,[5,7,4],3]     #in_layers_num, hidden_layers_num, out_layer_num
train_test_rate = 0.8
max_iter_num = 10
lr = 0.01
#mat_file_path = ''
#data_x, label_y = load_mat_file(mat_file_path)
import numpy as np
data_x = np.random.rand(20,8)
label_y = np.random.rand(20,3)
fnn = fnn_regression(fnn_net_structure)
dataTrain, dataTest = fnn_datasets(data_x, label_y, train_test_rate)
trained_fnn = fnn_train(fnn, dataTrain, max_iter_num, lr)
predict_value, true_value = fnn_predict(trained_fnn, dataTest)
regression_evaluation(predict_value, true_value)
