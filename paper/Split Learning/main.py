import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import click
from tqdm import tqdm
import gc
import os
import optuna

from losses import GroupedLoss, ImportanceLoss
from optimizers.drago import DRAGOForClasses
from optimizers.deshift import make_spectral_risk_measure, make_extremile_spectrum
from optimizers.large_scale_dro import RobustLoss
from optimizers.also import ALSO
from utils import set_global_seed, calculate_accuracy_at_k


class DualHeadModel(nn.Module):
    def __init__(self, num_classes_food, num_classes_flowers):
        super(DualHeadModel, self).__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.food_head = nn.Linear(512, num_classes_food)
        self.flowers_head = nn.Linear(512, num_classes_flowers)

    def forward(self, x, task=None):
        features = self.encoder(x)
        if task == 'food':
            return self.food_head(features)
        elif task == 'flowers':
            return self.flowers_head(features)
        return self.food_head(features), self.flowers_head(features)


def train_one_epoch(loader, task, model, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(images, task)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()


def train_both_datasets_one_epoch(loader_food, loader_flowers, model, optimizer, criterion, device, optim_mode, compute_weight_fn=None):
    model.train()
    if optim_mode in ('recover', 'adamw'):
        for (images_food, labels_food), (images_flowers, labels_flowers) in tqdm(zip(loader_food, loader_flowers), total=len(loader_flowers)):
            images_food, labels_food = images_food.to(device), labels_food.to(device)
            images_flowers, labels_flowers = images_flowers.to(device), labels_flowers.to(device)
            combined_images = torch.cat((images_food, images_flowers), dim=0)
            optimizer.zero_grad()
            outputs_food, outputs_flowers = model(combined_images)
            batch_size_food = images_food.size(0)
            food_predictions = outputs_food[:batch_size_food]
            flowers_predictions = outputs_flowers[batch_size_food:]
            loss_food = criterion(food_predictions, labels_food) / 2
            loss_flowers = criterion(flowers_predictions, labels_flowers) / 2
            total_loss = loss_food + loss_flowers
            total_loss.backward()
            optimizer.step()
    else:
        for (images_food, labels_food), (images_flowers, labels_flowers) in tqdm(zip(loader_food, loader_flowers), total=len(loader_flowers)):
            images_food, labels_food = images_food.to(device), labels_food.to(device)
            images_flowers, labels_flowers = images_flowers.to(device), labels_flowers.to(device)
            combined_images = torch.cat((images_food, images_flowers), dim=0)
            
            def closure(w=None, scale=None):
                optimizer.zero_grad()
                outputs_food, outputs_flowers = model(combined_images)
                batch_size_food = images_food.size(0)
                food_predictions = outputs_food[:batch_size_food]
                flowers_predictions = outputs_flowers[batch_size_food:]
                loss_food = criterion(food_predictions, labels_food)
                loss_flowers = criterion(flowers_predictions, labels_flowers)
                if not loss_food.size():
                    total_loss = (loss_food + loss_flowers)
                else:
                    total_loss = torch.concat((loss_food, loss_flowers))
                if w is None and compute_weight_fn is not None:
                    w = compute_weight_fn(total_loss)
                    scale = 1
                if w is not None and scale is not None:
                    total_loss = total_loss * scale
                    loss = (w * total_loss).sum()
                    loss.backward()
                    loss_log = total_loss.mean().item()
                else:
                    loss = total_loss.mean()
                    loss.backward()
                    loss_log = loss.item()
                return total_loss, loss_log
            closure.device = 'cuda'
            classes = torch.concat((labels_food, labels_flowers + 101)).to(device)
            if isinstance(optimizer, ALSO):
                optimizer.step(closure=closure, groups_indexes=classes)
            elif isinstance(optimizer, DRAGOForClasses):
                optimizer.step(closure=closure, classes=classes)
            elif isinstance(optimizer, AdamW):
                optimizer.step(closure=closure)
            

def evaluate(loader, task, model, criterion, device):
    model.eval()
    correct, total = 0, 0
    total_acc_at_3, total_acc_at_5 = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, task)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            acc_at_3 = calculate_accuracy_at_k(outputs, labels, k=3)
            acc_at_5 = calculate_accuracy_at_k(outputs, labels, k=5)
            total_acc_at_3 += acc_at_3 * images.size(0)
            total_acc_at_5 += acc_at_5 * images.size(0)
            del images, labels, outputs
            gc.collect()
    return correct / total, total_acc_at_3 / total, total_acc_at_5 / total


def get_criterion_and_optimizer(optim_mode, model, learning_rate, weight_decay, n_classes, batch_size, pilr, pidecay, tau, freq, size, reg, shift_cost, ndraws):
    compute_weights_fn, criterion, optimizer = None, None, None
    if 'adamw' in optim_mode:
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif 'importance' in optim_mode:
        criterion = ImportanceLoss(tau=tau)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif 'also' in optim_mode:
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = ALSO(model.parameters(), lr=learning_rate, weight_decay=weight_decay, n_groups=n_classes, batch_size=batch_size, mode='optimistic', pi_lr=pilr, pi_decay=pidecay)
    elif 'drago' in optim_mode:
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = DRAGOForClasses(model.parameters(), lr=learning_rate, weight_decay=weight_decay, n_classes=n_classes, batch_size=batch_size, pi_lr=pilr, pi_decay=pidecay, freq=freq)
    elif 'dro' in optim_mode:
        if 'largescale' in optim_mode:
            criterion = RobustLoss(
                    base_loss_fn=GroupedLoss(
                        base_loss=nn.CrossEntropyLoss(reduction='none'),
                        n_classes=n_classes,
                    ),
                    size=size,
                    reg=reg,
                    geometry='cvar'
                )
        else:
            criterion = GroupedLoss(
                    base_loss=nn.CrossEntropyLoss(reduction='none'),
                    n_classes=n_classes
                )
            penalty = "kl"
            spectrum = make_extremile_spectrum(batch_size, ndraws)
            compute_weights_fn = make_spectral_risk_measure(spectrum, penalty=penalty, shift_cost=shift_cost)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return criterion, optimizer, compute_weights_fn


def run(optim_mode, batch_size, num_epochs_food, num_epochs_joint, learning_rate, weight_decay, tau, pilr, pidecay, freq=0, size=0, reg=0, shift_cost=0, ndraws=0, seed=42, verbose=True):

    set_global_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    food101 = datasets.Food101(root="./data", split="train", transform=transform, download=True)
    flowers102 = datasets.Flowers102(root="./data", split="train", transform=transform, download=True)
    food101_test = datasets.Food101(root="./data", split="test", transform=transform, download=True)
    flowers102_test = datasets.Flowers102(root="./data", split="test", transform=transform, download=True)

    food101_loader = DataLoader(food101, batch_size=batch_size, shuffle=True, num_workers=4)
    flowers102_loader = DataLoader(flowers102, batch_size=batch_size, shuffle=True, num_workers=4)
    food101_test_loader = DataLoader(food101_test, batch_size=batch_size, shuffle=False, num_workers=4)
    flowers102_test_loader = DataLoader(flowers102_test, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes_flowers = 102

    model = DualHeadModel(num_classes_food=len(food101.classes), num_classes_flowers=num_classes_flowers)
    model = model.to(device)

    # Food101 only training in case if no valid checkpoints exists
    if not os.path.exists(f'checkpoints/adam_pretrain_epoch_{num_epochs_food}.pth'):
        print(f'No checkpoint adam_pretrain_epoch_{num_epochs_food}.pth found, so starting food101 only training first')
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        for epoch in tqdm(range(num_epochs_food)):
            train_one_epoch(food101_loader, 'food', model, optimizer, criterion, device)
            test_acc, test_acc_at_3, test_acc_at_5 = evaluate(food101_test_loader, 'food', model, criterion, device)
            torch.save(model.state_dict(), f'checkpoints/adam_pretrain_epoch_{epoch+1}.pth')
        print('Finished food101 only training')

    model.load_state_dict(torch.load(f'checkpoints/adam_pretrain_epoch_{num_epochs_food}.pth'))

    n_classes = len(food101.classes) + num_classes_flowers if '_classes' in optim_mode else 2

    criterion, optimizer, compute_weights_fn = get_criterion_and_optimizer(optim_mode, model, learning_rate, weight_decay, n_classes, batch_size, pilr, pidecay, tau, freq, size, reg, shift_cost, ndraws)

    if verbose:
        print(optimizer)
        print(criterion)

    food101_loader = DataLoader(food101, batch_size=batch_size // 2, shuffle=True, num_workers=2)
    flowers102_loader = DataLoader(flowers102, batch_size=batch_size // 2, shuffle=True, num_workers=2)

    logs = dict()
    for epoch in tqdm(range(num_epochs_joint)):
        train_both_datasets_one_epoch(food101_loader, flowers102_loader, model, optimizer, criterion, device, optim_mode, compute_weights_fn)
        food_test_acc, food_test_acc_at_3, food_test_acc_at_5 = evaluate(food101_test_loader, 'food', model, criterion, device)
        flowers_test_acc, flowers_test_acc_at_3, flowers_test_acc_at_5 = evaluate(flowers102_test_loader, 'flowers', model, criterion, device)
        logs[epoch] = (food_test_acc, food_test_acc_at_3, food_test_acc_at_5, flowers_test_acc, flowers_test_acc_at_3, flowers_test_acc_at_5)
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs_joint} [Joint Training] - "
                f"FOOD: Test Acc: {food_test_acc:.4f}, Test Acc@3: {food_test_acc_at_3:.4f}, Test Acc@5: {food_test_acc_at_5:.4f} | FLOWERS: "
                f"Test Loss: Test Acc: {flowers_test_acc:.4f}, Test Acc@3: {flowers_test_acc_at_3:.4f}, Test Acc@5: {flowers_test_acc_at_5:.4f}")
    return logs


def run_optuna(optim_mode, batch_size, num_epochs_food, num_epochs_joint):
    def objective(trial):
        pilr, pidecay, tau, freq, size, reg, shift_cost, ndraws = 0, 0, 0, 0, 0, 0, 0, 0
        if 'also' in optim_mode:
            learning_rate = trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
            pilr = trial.suggest_float('pilr', 1e-5, 1e-3, log=True)
            pidecay = trial.suggest_float('pidecay', 1e-3, 1, log=True)
        elif 'drago' in optim_mode:
            learning_rate = trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
            pilr = trial.suggest_float('pilr', 1e-5, 1e-3, log=True)
            pidecay = trial.suggest_float('pidecay', 1e-3, 1, log=True)
            freq = trial.suggest_int('freq', 1, 51, step=10)
        elif 'dro' in optim_mode:
            learning_rate = trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
            if 'largescale' in optim_mode:
                size = trial.suggest_float('size', 0.001, 0.999)
                reg = trial.suggest_float('reg', 0.001, 1)
            else:
                shift_cost = trial.suggest_float('shift_cost', 0.001, 1, log=True)
                ndraws = trial.suggest_float('ndraws', 0.001, 10, log=True)
        elif 'recover' in optim_mode:
            learning_rate = trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
            tau = trial.suggest_float('tau', 0.01, 3.0)
        elif 'adamw' in optim_mode:
            learning_rate = trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        else:
            raise ValueError(f"Unknown optim_mode: {optim_mode}")

        result = run(optim_mode, batch_size, num_epochs_food, num_epochs_joint, learning_rate, weight_decay, tau, pilr, pidecay, freq, size, reg, shift_cost, ndraws, seed=42, verbose=True)
        metrics = result[max(list(result.keys()))]
        return (metrics[2] + metrics[5]) / 2

    study = optuna.create_study(direction='maximize', study_name=f"test_optuna_{optim_mode}")
    study.optimize(objective, n_trials=50)
    
    with open(f'results/optuna_{optim_mode}', 'w') as f:
        f.write(str(study.best_params) + '\n' + str(study.best_value))
    
    return study.best_params, study.best_value


@click.command()
@click.option('--optim-mode')
@click.option('--batch_size', default=512, type=int)
@click.option('--num_epochs_food', help='Number of epochs for Food101-only training.', type=int)
@click.option('--num_epochs_joint', help='Number of epochs for joint training.', type=int)
@click.option('--learning_rate', default=1e-4, type=float)
@click.option('--weight-decay', default=0, type=float)
@click.option('--tau', default=1.5, type=float)
@click.option('--pilr', default=1e-5, type=float)
@click.option('--pidecay', default=1e-3, type=float)
@click.option('--optuna-mode', default=False, type=bool)
@click.option('--seeds', type=str)
@click.option('--verbose', default=True, type=bool)
def main(optim_mode, batch_size, num_epochs_food, num_epochs_joint, learning_rate, weight_decay, tau, pilr, pidecay, optuna_mode, seeds, verbose):
    if optuna_mode:
        run_optuna(optim_mode, batch_size, num_epochs_food, num_epochs_joint)
    else:
        seeds = eval(seeds)
        results = dict()
        for seed in seeds:
            result = run(optim_mode, batch_size, num_epochs_food, num_epochs_joint, learning_rate, weight_decay, tau, pilr, pidecay, seed=seed)
            results[(seed, optim_mode, learning_rate, pilr, pidecay)] = result
            if verbose:
                print(results)
            with open(f'results/{seed}_{optim_mode}_{learning_rate}', 'w') as f:
                f.write(str(results))


if __name__ == '__main__':
    main()
