import random
import os, sys

sys.path.insert(0, f"./../dataset/")
# sys.path.insert(0, f"../models")
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_loader import GetLoader
from model import LSTMModel  # Import the LSTM model instead of CNN
import numpy as np
from test import test
import time

# optuna hyperparam optimization

if __name__ == "__main__":
    source_dataset_name = "ESB1"
    target_dataset_name = "ESB2"

    features1 = [
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
    target1 = "EnergyConsumption"
    features2 = features1
    target2 = target1

    cuda = False
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 32
    n_epoch = 50
    input_size = len(features1) + 1  # features + previous timestep of label
    hidden_size = 32
    num_layers = 2

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    dataset_source = GetLoader(
        building_name="ESB",
        tower_number=1,
        features=features1,
        target=target1,
        season="summer",
        train_percentage=0.75,
        use_delta=False,
        shuffle_seed=42,
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=8
    )

    dataset_target = GetLoader(
        building_name="ESB",  # Adjust parameters accordingly
        tower_number=2,
        features=features2,
        target=target2,
        season="summer",
        train_percentage=0.75,
        use_delta=False,
        shuffle_seed=42,
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=8
    )

    # load model
    my_net = LSTMModel(input_size, hidden_size, num_layers)

    # setup optimizer
    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.MSELoss()  # MSE for regression
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    my_net = my_net.double()

    start_time = time.time()

    # training
    for epoch in range(n_epoch):
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0
        while i < len_dataloader:
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            # training model using source data
            s_data, s_label = next(data_source_iter)

            if cuda:
                s_data = s_data.cuda()
                s_label = s_label.cuda()
            s_data = s_data.clone().detach()

            class_output, domain_output = my_net(input_data=s_data, alpha=alpha)
            err_s_label = loss_class(class_output.squeeze(), s_label)
            err_s_domain = loss_domain(domain_output, torch.zeros(batch_size).long())

            # training model using target data
            t_data, _ = next(data_target_iter)

            if cuda:
                t_data = t_data.cuda()

            beta = 1  # TODO: hyperparam optimize
            _, domain_output = my_net(input_data=t_data, alpha=alpha)
            if (
                domain_output.size(0) < batch_size
            ):  # padding with ones so that sizes match
                padding = torch.ones(
                    batch_size - domain_output.size(0), *domain_output.shape[1:]
                )
                domain_output = torch.cat((domain_output, padding), dim=0)
            err_t_domain = loss_domain(domain_output, torch.ones(batch_size).long())
            err = err_t_domain + err_s_domain + beta * err_s_label
            err.backward()
            optimizer.step()

            i += 1

            print(
                "epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f"
                % (
                    epoch,
                    i,
                    len_dataloader,
                    err_s_label.cpu().data.numpy(),
                    err_s_domain.cpu().data.numpy(),
                    err_t_domain.cpu().data.numpy(),
                )
            )

        model_root = "./models_saved"
        torch.save(my_net, "{0}/model_epoch_{1}.pth".format(model_root, epoch))
        test(source_dataset_name, epoch)
        test(target_dataset_name, epoch)

    # Report the training time
    training_time = time.time() - start_time
    print(f"Training took {training_time:.2f} seconds")

    print("done")
