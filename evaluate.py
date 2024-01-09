import matplotlib.pyplot as plt

def draw_loss(train_losses,test_losses ):
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend()
    return
