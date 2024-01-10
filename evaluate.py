import matplotlib.pyplot as plt

def draw_loss(train_losses,valid_losses ):
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend()
    return
