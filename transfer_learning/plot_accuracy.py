import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import itertools

plotly.offline.init_notebook_mode()
# authenticate plotly here
def plot_accuracy(train_accuracy, valid_accuracy, step_size, file_name):
    data = []
    data.append(go.Scatter(
        y = np.array(train_accuracy),
        x = np.array(range(1, step_size).extend(range(step_size, step_size*len(train_accuracy)+1, step_size))),
        mode = 'lines',
        name = 'train_accuracy'
    ))
    data.append(go.Scatter(
        y = np.array(valid_accuracy),
        x = np.array(range(1, step_size).extend(range(step_size, step_size*len(train_accuracy)+1, step_size))),
        mode = 'lines',
        name = 'valid_accuracy'
    ))

    layout = go.Layout(
        xaxis=dict(
            title='iterations',
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
    '''
    figure=go.Figure(data=data,layout=layout)
    plotly.offline.iplot(figure, filename="mine", image="png")
    '''
    plotly.offline.plot({
        "data": data,
        "layout": layout
    })

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

