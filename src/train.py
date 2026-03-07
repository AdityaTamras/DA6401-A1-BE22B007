import argparse
import json
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from ann import NeuralNetwork, Optimizer
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, log_data_samples, one_hot

def load_model(model_path):
    data=np.load(model_path, allow_pickle=True).item()
    return data

def build_layer_dims(input_dim, hidden_dim, num_classes):
    return [input_dim] + hidden_dim + [num_classes]

def evaluate(model, X, y_true):
    Z_out, _ = model.forward(X)
    y_pred=np.argmax(Z_out, axis=0)
    acc=accuracy_score(y_true, y_pred)
    f1=f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1, y_pred

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Layer Perceptron")

    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Choose Dataset')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help='Choice of loss function')
    parser.add_argument('-o', '--optimizer', type=str,   choices=['sgd', 'momentum', 'nag', 'rmsprop'], default='sgd', help='Choice of optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='Weight decay for L2 regularization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128], help='Number of neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, choices=['sigmoid', 'tanh', 'relu'], default='sigmoid', help='Activation function for every hidden layer')
    parser.add_argument('-w_i', '--weight_init', dest='weight_init', type=str, choices=['random', 'xavier', 'zeros'], default='xavier', help='Technique to initialize weights')
    parser.add_argument('--model_path', type=str,  default='best_model.npy', help='Path to save/load best model weights')

    parser.add_argument('-w_p' '--wandb_project', dest='wandb_project', type=str, help='W&B Project ID')
    parser.add_argument('-e_n', '--exp_name', type=str, default='', help='Name of experiment')
    parser.add_argument('-lg', '--log_grads',  action='store_true', help='Logging per-neuron gradient norms for the first hidden layer')
    parser.add_argument('-la', '--log_acts',   action='store_true', help='Logging activation histograms for all hidden layers')
    parser.add_argument('-ldt', '--log_data_table', action='store_true', help='Logging a table of 5 sample images per class')
    parser.add_argument('-lc', '--log_confusion',  action='store_true', help='Logging confusion matrix at the end of training')

    args=parser.parse_args()

    hidden_sizes=args.hidden_size
    run_config=vars(args).copy()
    run_config['hidden_size']=' '.join(map(str, args.hidden_size))

    run=wandb.init(
        project=args.wandb_project, config=run_config, tags=[args.exp_name] if args.exp_name else []
    )

    cfg=wandb.config

    raw_hidden=cfg.hidden_size
    if isinstance(raw_hidden, str):
        hidden_sizes=list(map(int, raw_hidden.split()))
    elif isinstance(raw_hidden, (list, tuple)):
        hidden_sizes=list(map(int, raw_hidden))
    else:
        hidden_sizes=[int(raw_hidden)]

    X_train_raw, y_train_raw, X_test, y_test=load_dataset(args.dataset)

    if args.log_data_table:
        if cfg.dataset=='mnist':
            from keras.datasets import mnist as _ds
        else:
            from keras.datasets import fashion_mnist as _ds
        (X_raw, y_raw), _ = _ds.load_data()
        log_data_samples(X_raw, y_raw, args.dataset)

    y_train=one_hot(y_train_raw)

    layer_dims=build_layer_dims(X_train_raw.shape[0], hidden_sizes, y_train.shape[0])

    model=NeuralNetwork(layer_dims=layer_dims, weight_init=cfg.weight_init, activation_function=cfg.activation)
    optim=Optimizer(method=cfg.optimizer, lr=cfg.learning_rate, wd=cfg.weight_decay)

    print(f"[{cfg.exp_name or 'train'}], dataset={cfg.dataset}, layers={layer_dims}, optim={cfg.optimizer},  activation={cfg.activation}, loss={cfg.loss}, lr={cfg.learning_rate},  init={cfg.weight_init},  epochs={cfg.epochs}")

    N=X_train_raw.shape[1]
    best_f1_score=-1.0
    best_weights=None
    global_step=0

    for epoch in range(cfg.epochs):
        epoch_loss=0
        num_batches=0

        perm=np.random.permutation(N)
        X_train=X_train_raw[:, perm]
        y_train_s=y_train[:, perm]

        for i in range(0, N, args.batch_size):
            X_batch=X_train[:, i:i+args.batch_size]
            y_batch=y_train_s[:, i:i+args.batch_size]
            Z_out, cache = model.forward(X_batch)
            loss=model.compute_loss(Z_out, y_batch, args.loss)
            grads=model.backward(Z_out, y_batch, args.loss, cache)
            optim.update_parameters(model.init_params, grads)
            epoch_loss+=loss
            num_batches+=1
            global_step+=1
        
            if args.log_grads and global_step<=50:
                dw1=grads['dW0']
                grad_log={'grad_step': global_step}
                for neuron_idx in range(min(5, dw1.shape[0])):
                    grad_log[f'GradNorm/Layer1_Neuron{neuron_idx+1}'] = float(np.linalg.norm(dw1[neuron_idx]))
                grad_log['GradNorm/Layer1_Overall'] = float(np.linalg.norm(dw1))
                wandb.log(grad_log, step=global_step)

        avg_loss=epoch_loss/num_batches
        
        train_subset_idx=np.random.choice(N, size=min(5000, N), replace=False)
        train_acc, train_f1, _ = evaluate(model, X_train_raw[:, train_subset_idx], y_train_raw[train_subset_idx])
        val_acc, val_f1, y_pred = evaluate(model, X_test, y_test)

        log_dict={
            'epoch': epoch+1,
            'train_loss': avg_loss,
            'train_accuracy': train_acc,
            'train_f1_score': train_f1,
            'val_accuracy': val_acc,
            'val_f1_score': val_f1
        }

        if args.log_acts:
            for layer_idx, act_vals in enumerate(model.hidden_activations):
                log_dict[f'Activations/Layer_{layer_idx+1}'] = wandb.Histogram(act_vals)

        if args.log_grads:
            for l_idx, gnorm in enumerate(model.layer_grad_norms):
                log_dict[f'GradNorm/Layer{l_idx+1}_EpochEnd'] = gnorm

        wandb.log(log_dict)

        print(f"Epoch [{epoch+1:>3}/{cfg.epochs}], loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")



        if val_f1>best_f1_score:
            best_f1_score=val_f1
            best_weights=model.get_weights()
            np.save(args.model_path, np.array(best_weights, dtype=object), allow_pickle=True)
            with open("best_config.json", "w") as f:
                json.dump(vars(args), f, indent=4)

    if best_weights is not None:
        model.set_weights(best_weights)
    
    final_val_acc, final_val_f1, y_pred_best=evaluate(model, X_test, y_test)
    wandb.summary['best_val_accuracy']=final_val_acc
    wandb.summary['best_val_f1']=final_val_f1

    if args.log_confusion:
        if cfg.dataset=='mnist':
            class_names=[str(i) for i in range(10)]
        else:
            class_names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        wandb.log({
            'Confusion Matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_test.tolist(), preds=y_pred_best.tolist(), class_names=class_names)
        })

        misclassified_table=wandb.Table(columns=['Index', 'True Label', 'Predicted Label', 'Image'])
        mis_idx=np.where(y_pred_best != y_test)[0]
        np.random.shuffle(mis_idx)
        X_test_img=X_test.T.reshape(-1, 28, 28)
        for idx in mis_idx[:50]:
            misclassified_table.add_data(int(idx), class_names[y_test[idx]], class_names[y_pred_best[idx]], wandb.Image(X_test_img[idx]))
        wandb.log({'Error Analysis/Misclassified Samples': misclassified_table})

    run.finish()

if __name__ == '__main__':
    main()
