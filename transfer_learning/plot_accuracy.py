import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.interpolate import spline

plotly.offline.init_notebook_mode()

def plot_accuracy_num_train_samples(accuracy, min_samples, max_samples, step_size, name):
    #plots a smooth curve
    data = []
    x = np.array(range(min_samples, max_samples, step_size))
    x_new = np.linspace(x.min(),x.max(),200)
    y = np.array(accuracy)
    y_new = spline(x,y,x_new)
    '''
    data.append(go.Scatter(
        y = np.array(accuracy),
        x = np.array(range(min_samples, max_samples, step_size)),
        mode = 'lines',
        name = name
    ))

    layout = go.Layout(
        xaxis=dict(
            title='num_of_train_samples',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='accuracy',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    plotly.offline.plot({
        "data": data,
        "layout": layout
    })
    '''
    plt.ylabel('Test accuracy')
    plt.xlabel('Number of training samples')
    plt.plot(x_new,y_new)
    plt.show()

# authenticate plotly here
def plot_accuracy(data, x_axis_title, y_axis_title, filename):

    layout = go.Layout(
        xaxis=dict(
            title=x_axis_title,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=y_axis_title,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
   
    plotly.offline.plot(data, filename=filename,config = layout)

def plot_confusion_matrix(cm, classes,
                          file_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,):
    plt.figure()
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)

