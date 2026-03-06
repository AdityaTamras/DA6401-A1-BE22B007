import argparse
import json
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ann import NeuralNetwork
from utils.data_loader import load_dataset, FASHION_MNIST_LABELS

def load_model(model_path):
    data=np.load(model_path, allow_pickle=True).item()
    return data

def main():
    parser = argparse.ArgumentParser(description="Run inference with a saved MLP checkpoint")

    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Choose Dataset')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help='Choice of loss function')
    parser.add_argument('-o', '--optimizer', type=str,   choices=['sgd', 'momentum', 'nag', 'rmsprop'], default='sgd', help='Choice of optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='Weight decay for L2 regularization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=str, nargs='+', required=True, help='Number of neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, choices=['sigmoid', 'tanh', 'relu'], default='sigmoid', help='Activation function for every hidden layer')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'xavier'], default='random', help='Technique to initialize weights')
    parser.add_argument('--model_path', type=str,  default='best_model.npy', help='Path to save/load best model weights')

    parser.add_argument('-w_p' '--wandb_project', dest='wandb_project', type=str, help='W&B Project ID')
    parser.add_argument('-e_n', '--exp_name', type=str, default='', help='Name of experiment')
    parser.add_argument('-lg', '--log_grads',  action='store_true', help='Logging per-neuron gradient norms for the first hidden layer')
    parser.add_argument('-la', '--log_acts',   action='store_true', help='Logging activation histograms for all hidden layers')
    parser.add_argument('-ldt', '--log_data_table', action='store_true', help='Logging a table of 5 sample images per class')
    parser.add_argument('-lc', '--log_confusion',  action='store_true', help='Logging confusion matrix at the end of training')

    args=parser.parse_args()

    run_config=vars(args).copy()
    run_config['hidden_size']=' '.join(args.hidden_size)

    run=wandb.init(
        project=args.wandb_project, config=run_config, tags=[args.exp_name] if args.exp_name else []
    )

    cfg=wandb.config

    _, _, X_test, y_test = load_dataset(args.dataset)

    hidden_sizes=list(map(int, args.hidden_size))
    layer_dims=[X_test.shape[0]] + hidden_sizes + [10]

    model = NeuralNetwork(layer_dims=layer_dims, weight_init=args.weight_init, activation_function=args.activation)

    print(f"Loading weights from {args.model_path} ...")
    weights=load_model(args.model_path)
    model.set_weights(weights)

    Z_out, _ = model.forward(X_test)
    y_pred=np.argmax(Z_out, axis=0)
    accuracy=accuracy_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred, average='macro')
    precision=precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall=recall_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n── Test Results ──────────────────────────")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"──────────────────────────────────────────")

    if args.log_confusion:
        if cfg.dataset=='mnist':
            class_names=[str(i) for i in range(10)]
        else:
            class_names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        wandb.log({
            'Confusion Matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_test.tolist(), preds=y_pred.tolist(), class_names=class_names)
        })

        misclassified_table=wandb.Table(columns=['Index', 'True Label', 'Predicted Label', 'Image'])
        mis_idx=np.where(y_pred!=y_test)[0]
        np.random.shuffle(mis_idx)
        X_test_img=X_test.T.reshape(-1, 28, 28)
        for idx in mis_idx[:50]:
            misclassified_table.add_data(int(idx), class_names[y_test[idx]], class_names[y_pred[idx]], wandb.Image(X_test_img[idx]))
        wandb.log({'Error Analysis/Misclassified Samples': misclassified_table})

if __name__ == '__main__':
    main()