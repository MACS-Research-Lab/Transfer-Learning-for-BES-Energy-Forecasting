import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from data_loader import GetLoader
from torch.nn.functional import mse_loss


def test(dataset_name, epoch):
    assert dataset_name in ["ESB1", "ESB2"]

    model_root = os.path.join(".", "models_saved")

    cuda = False
    cudnn.benchmark = True
    batch_size = 32
    alpha = 0

    features = [
        "FlowEvap",
        "PerHumidity",
        "TempAmbient",
        "TempCondIn",
        "TempCondOut",
        "TempEvapIn",
        "TempEvapOut",
        "TempWetBulb",
        "PerFreqConP",
        "Tonnage",
        "DayOfWeek",
        "HourOfDay",
        "PerFreqFan",
    ]
    target = "EnergyConsumption"

    """load data"""
    if dataset_name == "ESB1":
        dataset = GetLoader(
            building_name="ESB",  # Adjust parameters accordingly
            tower_number=1,
            features=features,
            target=target,
            train_percentage=0.75,
            use_delta=False,
            shuffle_seed=42,
        )
    else:
        dataset = GetLoader(
            building_name="ESB",
            tower_number=2,
            features=features,
            target=target,
            train_percentage=0.75,
            use_delta=False,
            shuffle_seed=42,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    """ testing """
    my_net = torch.load(os.path.join(model_root, "model_epoch_" + str(epoch) + ".pth"))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    mse_total = 0.0

    for data_target in data_target_iter:
        t_data, t_label = data_target

        if cuda:
            t_data = t_data.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_data, alpha=alpha)
        mse_total += mse_loss(class_output.squeeze(), t_label).item()

    average_mse = mse_total / len_dataloader

    print(
        "epoch: %d, Mean Squared Error on the %s dataset: %f"
        % (epoch, dataset_name, average_mse)
    )
