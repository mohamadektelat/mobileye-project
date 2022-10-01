try:
    from Configurations import *
except ImportError:
    print("Need to fix the installation")
    raise


def train():
    epoch_index = []
    loss_value = []

    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)
            loss = criterion(scores, targets.unsqueeze(-1).float())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        epoch_index.append(epoch + 1)
        loss_value.append(sum(losses) / len(losses))
        print(f'Cost at epoch{epoch} is {sum(losses) / len(losses)}')


def check_accuracy(loader, model):
    """
    This function is responsible for calculating accuracy.
    :param loader: The dataset loader.
    :param model: The trained model.
    :return: None.
    """
    num_correct = 0
    num_samples = 0
    model.eval()
    # set all the grad flags to be zero
    with torch.no_grad():
        # data, targets
        for data, target in loader:
            data = data.to(device=device)
            target = target.to(device=device)

            score = model(data)
            _, predictions = score.max(1)

            num_correct += (predictions == target).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100}")

    model.train()


def plot_fig(x_coordinates, y_coordinates):
    """
    This function is responsible for drawing the loss figure.
    :param x_coordinates: List for the epoch index.
    :param y_coordinates: List for the loss value.
    :return: None.
    """
    plt.plot(x_coordinates, y_coordinates)
    plt.xlabel('x - Epoch Index')
    plt.ylabel('y - Loss Value')
    plt.title('Loss Graph')
    plt.savefig('Loss_Graph.jpg')
    plt.show()


def save_model(model):
    """
    This function is responsible for saving the trained model.
    :param model: The model trained.
    :return: None.
    """
    file = "model.pth"
    torch.save(model, file)


print("Checking accuracy on Train Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
