import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv

def main():
    path = sys.argv[1]
    outputcsv = sys.argv[2] # this is without the .csv extension, automatically added

    results = torch.load(path)

    val_acc = torch.FloatTensor(results['tracker']['val_acc'])
    val_acc = val_acc.mean(dim=1).numpy()

    train_acc = torch.FloatTensor(results['tracker']['train_acc'])
    train_acc = train_acc.mean(dim=1).numpy()

    train_loss = torch.FloatTensor(results['tracker']['train_loss'])
    train_loss = train_loss.mean(dim=1).numpy()

    val_loss = torch.FloatTensor(results['tracker']['val_loss'])
    val_loss = val_loss.mean(dim=1).numpy()


    # save to .csv
    with open(outputcsv + '.csv', mode='w') as out:
        outw = csv.writer(out, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        outw.writerow(['Epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss'])


        for i in range(len(val_acc)):
            outw.writerow(['{}'.format(i+1), train_acc[i], val_acc[i], train_loss[i],
                           val_loss[i]])


    plt.figure()
    plt.plot(val_acc, label = "Validation Accuracy")
    plt.plot(train_acc, label = "Training Accuracy")
    plt.legend()
    plt.title("Plot of Accuracy against Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(outputcsv + '-acc.png')

    # clear figure and plot the loss
    plt.clf()
    plt.plot(val_loss, label = "Validation Loss")
    plt.plot(train_loss, label = "Training Loss")
    plt.legend()
    plt.title("Plot of Loss against Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(outputcsv + '-loss.png')

if __name__ == '__main__':
    main()
