"""
This file contains various plotting functions for network predictions, testing, and debugging
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import seaborn as sns; sns.set()
import utils.logging as logging
import utils
from PyQt5.QtWidgets import (QFileDialog, QAbstractItemView, QListView,
                             QTreeView, QApplication, QDialog)

def plot_all(logit1, tr1, logit2 = None, tr2 = None, model = None, index = 0, xmin=20, xmax=40, num_points=1001,
               num_osc = 10, title=None, figsize=[12, 9], y_axis='Test Variable', label_y1='Pred 1', label_y2='Pred 2'):
    """
    Function to plot various predicted and ground truth spectra, as well as Lorentzian parameters
    :param logit1, logit2:  Predicted spectra, typically real or imaginary part of complex s-parameters (r,t)
    :param tr1, tr2:  Ground truth spectra
    :param xmin, xmax: Spectral range plotted
    :param num_osc: Number of Lorentzian oscillators, for building the parameter table
    :param title: The title of the plot, default None
    :param figsize: The figure size of the plot
    :param y_axis: Name of spectrum being plotted
    :return: The identifier of the figure
    """
    # Make the frequency points
    frequency = xmin + (xmax - xmin) / num_points * np.arange(num_points)

    f1 = plt.figure(figsize=figsize)
    ax11 = plt.subplot2grid((3, 3), (0, 0))
    ax12 = plt.subplot2grid((3, 3), (0, 1))
    ax13 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax21 = plt.subplot2grid((3, 3), (1, 0))
    ax22 = plt.subplot2grid((3, 3), (1, 1))

    ax11.plot(frequency, logit1, label=label_y1)
    ax11.plot(frequency, tr1, label="Truth 1")

    if logit2 is not None:                              # If a second predicted spectrum is provided, plot it
        ax11.plot(frequency, logit2, label=label_y2)
        ax11.plot(frequency, tr2, label="Truth 2")

    # Various optical constants or other physics quantities included here
    ax12.plot(frequency, model.n_out[index].real.cpu().data.numpy(), label="n")
    ax12.plot(frequency, model.n_out[index].imag.cpu().data.numpy(), color='r', label="k")
    ax21.plot(frequency, model.eps_out[index].real.cpu().data.numpy(), label="e1")
    ax21.plot(frequency, model.eps_out[index].imag.cpu().data.numpy(), color='r', label="e2")
    ax22.plot(frequency, model.mu_out[index].real.cpu().data.numpy(), label="mu1")
    ax22.plot(frequency, model.mu_out[index].imag.cpu().data.numpy(), color='r', label="mu2")
    # ax21.plot(frequency, model.theta_out[index].real.cpu().data.numpy(), label="theta_re")
    # ax21.plot(frequency, model.theta_out[index].imag.cpu().data.numpy(), color='r', label="theta_im")
    # ax22.plot(frequency, model.adv_out[index].real.cpu().data.numpy(), label="adv_re")
    # ax22.plot(frequency, model.adv_out[index].imag.cpu().data.numpy(), color='r', label="adv_im")

    ax11.legend()
    ax12.legend()
    ax21.legend()
    ax22.legend()
    ax22.set_xlabel("Frequency (THz)")
    ax11.set_ylabel("Test Variable")
    if title is not None:
        plt.title(title)

    # This plotting function also includes a table containing the Lorentzian parameters

    at = AnchoredText("eps_inf: " + str(np.round(1+model.eps_params_out[3][index].cpu().data.numpy(), 3)) + \
                        ", mu_inf: " + str(np.round(1+model.mu_params_out[3][index].cpu().data.numpy(), 3)) + \
                        ", d: " + str(np.round(model.d_out[index].cpu().data.numpy(), 3)),
                      prop=dict(size=10), frameon=True,
                      loc='lower left'
                      )

    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    at2 = AnchoredText("Geom: " + str(np.round(model.geom[index].cpu().data.numpy(), 4)),
                      prop=dict(size=10), frameon=True,
                      loc='upper left'
                      )
    at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

    ax13.add_artist(at)
    ax13.add_artist(at2)
    columns = ('eps w0', 'eps wp', 'eps g', 'mu w0', 'mu wp', 'mu g')
    table_data = np.empty((6, num_osc))
    table_data[0, :] = model.eps_params_out[0][index].cpu().data.numpy()
    table_data[1, :] = model.eps_params_out[1][index].cpu().data.numpy()
    table_data[2, :] = model.eps_params_out[2][index].cpu().data.numpy()
    table_data[3, :] = model.mu_params_out[0][index].cpu().data.numpy()
    table_data[4, :] = model.mu_params_out[1][index].cpu().data.numpy()
    table_data[5, :] = model.mu_params_out[2][index].cpu().data.numpy()

    table_data = np.round(table_data, 3)
    ax13.table(cellText=np.transpose(table_data), colLabels=columns, loc='center',fontsize=72)
    ax13.axis('off')
    ax13.grid(b=None)
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    # ax13.figure.set_size_inches(10, 5)

    return f1

def plot_complex(logit1, tr1, logit2 = None, tr2 = None, xmin=20, xmax=40, num_points=1001, title=None, figsize=[10, 5],
                    y_axis='Test Variable', label_y1='T', label_y2='R'):
    """
    Function to plot either a complex quantity (eg. r,t) or R,T spectra
    :param logit1, logit2:  Predicted spectra, typically real or imaginary part of complex s-parameters (r,t)
    :param tr1, tr2:  Ground truth spectra
    :param xmin, xmax: Spectral range plotted
    :param title: The title of the plot, default None
    :param figsize: The figure size of the plot
    :param y_axis: Name of spectrum being plotted
    :return: The identifier of the figure
    """
    # Make the frequency points
    frequency = xmin + (xmax - xmin) / num_points * np.arange(num_points)
    mse_loss = np.mean((logit1 - tr1) ** 2)
    if logit2 is not None:
        f, [ax1,ax2] = plt.subplots(1,2,figsize=figsize)
        ax1.plot(frequency, logit1, label=label_y1)
        ax1.plot(frequency, tr1, label="Truth")
        ax2.plot(frequency, logit2, label=label_y2)
        ax2.plot(frequency, tr2, label="Truth")
    else:
        f, ax1 = plt.subplots(1,1,figsize=figsize)
        plt.plot(frequency, logit1, label=label_y1)
        plt.plot(frequency, tr1, label="Truth")
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel(y_axis)
    plt.grid(b=None)
    if title is not None:
        plt.title(title)

    # Plots MSE of logit1
    at = AnchoredText("MSE: " + str(np.round(mse_loss, 6)),
                      prop=dict(size=15), frameon=True,
                      loc='lower left',
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    return f

def plot_weights_3D(data, dim, figsize=[10, 5]):
    """
    Takes in the weights or gradients of a layer and converts it to a square N x N surface plot
    :param: dim: Dimension N of the square surface plot
    :return: The identifier of the figure
    """
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection='3d', proj_type='ortho')
    ax2 = fig.add_subplot(122)

    xx, yy = np.meshgrid(np.linspace(0, dim, dim), np.linspace(0, dim, dim))
    cmp = plt.get_cmap('viridis')

    ax1.plot_surface(xx, yy, data, cmap=cmp)
    ax1.view_init(10, -45)

    c2 = ax2.imshow(data, cmap=cmp)
    plt.colorbar(c2, fraction=0.03)
    plt.grid(b=None)

    return fig

def plotMSELossDistrib(pred, truth):
    """
    Plots the MSE distribution for a given prediction/truth batch or file
    :return: The identifier of the figure
    """
    # mae, mse = compare_truth_pred(pred_file, truth_file)
    # mae = np.mean(np.abs(pred - truth), axis=1)
    mse = np.mean(np.square(pred - truth), axis=1)
    # mse = loss
    f = plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Validation Loss')
    plt.ylabel('Count')
    plt.suptitle('Model (Avg MSE={:.4e})'.format(np.mean(mse)))
    # plt.savefig(os.path.join(os.path.abspath(''), 'models',
    #                          'MSEdistrib_{}.png'.format(flags.model_name)))
    return f

def plotMSELossDistrib_eval(pred_file, truth_file, flags):
    mae, mse = compare_truth_pred(pred_file, truth_file)
    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('cnt')
    plt.suptitle('(Avg MSE={:.4e})'.format(np.mean(mse)))
    eval_model_str = flags.eval_model.replace('/','_')
    plt.savefig(os.path.join(os.path.abspath(''), 'eval',
                         '{}.png'.format(eval_model_str)))
    print('(Avg MSE={:.4e})'.format(np.mean(mse)))

class HMpoint(object):
    """
    This is a HeatMap point class where each object is a point in the heat map
    properties:
    1. BV_loss: best_validation_loss of this run
    2. feature_1: feature_1 value
    3. feature_2: feature_2 value, none is there is no feature 2
    """
    def __init__(self, bv_loss, f1, f2 = None, f1_name = 'f1', f2_name = 'f2'):
        self.bv_loss = bv_loss
        self.feature_1 = f1
        self.feature_2 = f2
        self.f1_name = f1_name
        self.f2_name = f2_name
        #print(type(f1))
    def to_dict(self):
        return {
            self.f1_name: self.feature_1,
            self.f2_name: self.feature_2,
            self.bv_loss: self.bv_loss
        }

def HeatMapBVL(plot_x_name, plot_y_name, title,  save_name='HeatMap.png', HeatMap_dir = 'HeatMap',
                feature_1_name=None, feature_2_name=None,
                heat_value_name = 'best_validation_loss'):
    """
    Plotting a HeatMap of the Best Validation Loss for a batch of hypersweeping
    First, copy those models to a folder called "HeatMap"
    Algorithm: Loop through the directory using os.look and find the parameters.txt files that stores the
    :param HeatMap_dir: The directory where the checkpoint folders containing the parameters.txt files are located
    :param feature_1_name: The name of the first feature that you would like to plot on the feature map
    :param feature_2_name: If you only want to draw the heatmap using 1 single dimension, just leave it as None
    """
    one_dimension_flag = False          #indication flag of whether it is a 1d or 2d plot to plot
    #Check the data integrity 
    if (feature_1_name == None):
        print("Please specify the feature that you want to plot the heatmap");
        return
    if (feature_2_name == None):
        one_dimension_flag = True
        print("You are plotting feature map with only one feature, plotting loss curve instead")

    #Get all the parameters.txt running related data and make HMpoint objects
    HMpoint_list = []
    df_list = []                        #make a list of data frame for further use
    for subdir, dirs, files in os.walk(HeatMap_dir):
        for file_name in files:
             if (file_name == 'parameters.txt'):
                file_path = os.path.join(subdir, file_name) #Get the file relative path from 
                # df = pd.read_csv(file_path, index_col=0)
                flag = logging.load_flags(subdir)
                flag_dict = vars(flag)
                df = pd.DataFrame()
                for k in flag_dict:
                    df[k] = pd.Series(str(flag_dict[k]), index=[0])
                print(df)
                if (one_dimension_flag):
                    #print(df[[heat_value_name, feature_1_name]])
                    #print(df[heat_value_name][0])
                    #print(df[heat_value_name].iloc[0])
                    df_list.append(df[[heat_value_name, feature_1_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(str(df[feature_1_name][0])), 
                                                f1_name = feature_1_name))
                else:
                    if feature_2_name == 'linear_unit':                         # If comparing different linear units
                        df['linear_unit'] = eval(df[feature_1_name][0])[1]
                        df['best_validation_loss'] = get_bvl(file_path)
                    df_list.append(df[[heat_value_name, feature_1_name, feature_2_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]),eval(str(df[feature_1_name][0])),
                                                eval(str(df[feature_2_name][0])), feature_1_name, feature_2_name))
    
    print(df_list)
    #Concatenate all the dfs into a single aggregate one for 2 dimensional usee
    df_aggregate = pd.concat(df_list, ignore_index = True, sort = False)
    #print(df_aggregate[heat_value_name])
    #print(type(df_aggregate[heat_value_name]))
    df_aggregate.astype({heat_value_name: 'float'})
    #print(type(df_aggregate[heat_value_name]))
    #df_aggregate = df_aggregate.reset_index()
    print("before transformation:", df_aggregate)
    [h, w] = df_aggregate.shape
    for i in range(h):
        for j in range(w):
            if isinstance(df_aggregate.iloc[i,j], str) and (isinstance(eval(df_aggregate.iloc[i,j]), list)):
                # print("This is a list!")
                df_aggregate.iloc[i,j] = len(eval(df_aggregate.iloc[i,j]))

    print("after transformation:",df_aggregate)
    
    #Change the feature if it is a tuple, change to length of it
    for cnt, point in enumerate(HMpoint_list):
        print("For point {} , it has {} loss, {} for feature 1 and {} for feature 2".format(cnt, 
                                                                point.bv_loss, point.feature_1, point.feature_2))
        assert(isinstance(point.bv_loss, float))        #make sure this is a floating number
        if (isinstance(point.feature_1, tuple)):
            point.feature_1 = len(point.feature_1)
        if (isinstance(point.feature_2, tuple)):
            point.feature_2 = len(point.feature_2)

    f = plt.figure()
    #After we get the full list of HMpoint object, we can start drawing 
    if (feature_2_name == None):
        print("plotting 1 dimension HeatMap (which is actually a line)")
        HMpoint_list_sorted = sorted(HMpoint_list, key = lambda x: x.feature_1)
        #Get the 2 lists of plot
        bv_loss_list = []
        feature_1_list = []
        for point in HMpoint_list_sorted:
            bv_loss_list.append(point.bv_loss)
            feature_1_list.append(point.feature_1)
        print("bv_loss_list:", bv_loss_list)
        print("feature_1_list:",feature_1_list)
        #start plotting
        plt.plot(feature_1_list, bv_loss_list,'o-')
        np.savetxt('loss_list.txt',bv_loss_list,delimiter='\t')
    else: #Or this is a 2 dimension HeatMap
        print("plotting 2 dimension HeatMap")
        #point_df = pd.DataFrame.from_records([point.to_dict() for point in HMpoint_list])
        df_aggregate = df_aggregate.reset_index()
        df_aggregate.sort_values(feature_1_name, axis=0, inplace=True)
        df_aggregate.sort_values(feature_2_name, axis=0, inplace=True)
        df_aggregate.sort_values(heat_value_name, axis=0, inplace=True)
        print("before dropping", df_aggregate)
        df_aggregate = df_aggregate.drop_duplicates(subset=[feature_1_name, feature_2_name], keep='first')
        print("after dropping", df_aggregate)
        point_df_pivot = df_aggregate.reset_index().pivot(index=feature_1_name, columns=feature_2_name, values=heat_value_name).astype(float)
        point_df_pivot = point_df_pivot.rename({'5': '05'}, axis=1)
        point_df_pivot = point_df_pivot.reindex(sorted(point_df_pivot.columns), axis=1)
        print("pivot=")
        csvname = HeatMap_dir + 'pivoted.csv'
        point_df_pivot.to_csv(csvname)
        print(point_df_pivot)
        sns.heatmap(point_df_pivot, cmap = "YlGnBu")
    plt.xlabel(plot_y_name)                 # Note that the pivot gives reversing labels
    plt.ylabel(plot_x_name)                 # Note that the pivot gives reversing labels
    plt.title(title)
    plt.savefig(save_name)

def get_bvl(file_path):
    """
    This is a helper function for 0119 usage where the bvl is not recorded in the pickled object but in .txt file and needs this funciton to retrieve it
    """
    df = pd.read_csv(file_path, delimiter=',')
    bvl = 0
    for col in df:
        if 'best_validation_loss' in col:
            print(col)
            strlist = col.split(':')
            bvl = eval(strlist[1][1:-2])
    if bvl == 0:
        print("Error! We did not found a bvl in .txt.file")
    else:
        return float(bvl)


def plot_loss_folder_comparison():
    qapp = QApplication(sys.argv)
    print('Get Directories now')
    dirs = utils.getExistingDirectories()
    if dirs.exec_() == utils.QDialog.Accepted:
        folder_paths = dirs.selectedFiles()
        folder_names = [value.split('/')[-1] for c, value in enumerate(folder_paths)]
        # losses = np.empty((len(folder_names)))
        df = pd.DataFrame(columns=['Loss','Model'])

        for i in range(len(folder_names)):
            file_path = folder_paths[i] + '/parameters.txt'
            loss = get_bvl(file_path)
            model = '_'.join(folder_names[i].split('_')[1:-1])
            df = df.append({'Loss': loss, 'Model': model}, ignore_index=True)

        # print(df)
        # curr_col = '_'.join(folder_names[0].split('_')[:-1])
        # loss = get_bvl(folder_paths[0] + '/parameters.txt')
        # data = np.array(loss)
        # for i in range(1,len(folder_names)):
        #     file_path = folder_paths[i] + '/parameters.txt'
        #     if (curr_col != '_'.join(folder_names[i].split('_')[:-1])):
        #         col = '_'.join(folder_names[i-1].split('_')[:-1])
        #         df[col] = pd.Series(data)
        #         loss = get_bvl(file_path)
        #         data = np.array(loss)
        #         curr_col = '_'.join(folder_names[i].split('_')[:-1])
        #     else:
        #         loss = get_bvl(file_path)
        #         data = np.append(data, loss)
        #
        # col = '_'.join(folder_names[-1].split('_')[:-1])
        # df[col] = pd.Series(data)

        # return df
        plt.switch_backend('Qt5Agg')
        fig, ax = plt.subplots(num=2, figsize=(10,5))

        sns.set(style="whitegrid", color_codes=True)
        ax = sns.swarmplot(x="Model", y="Loss", data=df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Validation Loss Comparison', fontsize=14)
        # print(matplotlib.get_backend())
        # plt.tight_layout()
        plt.figure(num=2)
        plt.show()

        plt.savefig('C:/Users/labuser/mlmOK_Pytorch/Loss_Comparison.png', bbox_inches='tight')
        # plt.savefig('C:/Users/labuser/mlmOK_Pytorch/Loss_Comparison.png')

        return df

def compare_truth_pred(pred_file, truth_file, cut_off_outlier_thres=None, quiet_mode=False):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    if isinstance(pred_file, str):  # If input is a file name (original set up)
        pred = np.loadtxt(pred_file, delimiter=' ')
        truth = np.loadtxt(truth_file, delimiter=' ')
    elif isinstance(pred_file, np.ndarray):
        pred = pred_file
        truth = truth_file
    else:
        print('In the compare_truth_pred function, your input pred and truth is neither a file nor a numpy array')
    if not quiet_mode:
        print("in compare truth pred function in eval_help package, your shape of pred file is", np.shape(pred))
    if len(np.shape(pred)) == 1:
        # Due to Ballistics dataset gives some non-real results (labelled -999)
        valid_index = pred != -999
        if (np.sum(valid_index) != len(valid_index)) and not quiet_mode:
            print("Your dataset should be ballistics and there are non-valid points in your prediction!")
            print('number of non-valid points is {}'.format(len(valid_index) - np.sum(valid_index)))
        pred = pred[valid_index]
        truth = truth[valid_index]
        # This is for the edge case of ballistic, where y value is 1 dimensional which cause dimension problem
        pred = np.reshape(pred, [-1, 1])
        truth = np.reshape(truth, [-1, 1])
    mae = np.mean(np.abs(pred - truth), axis=1)
    mse = np.mean(np.square(pred - truth), axis=1)

    if cut_off_outlier_thres is not None:
        mse = mse[mse < cut_off_outlier_thres]
        mae = mae[mae < cut_off_outlier_thres]

    return mae, mse