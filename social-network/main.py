import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN

criterion = nn.CrossEntropyLoss()

ALLAcc = 0
FALLAcc = 0
tagset = set([])
#for g in train_graphs:
#    tagset = tagset.union(set(g.node_tags))
#tagset = list(tagset)
def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    #batch_graph = [train_graphs[idx] for idx in selected_idx]
    #tagset = set([])
    #for g in train_graphs:
    #    tagset = tagset.union(set(g.node_tags))
    #tagset = list(tagset)
    #tag2index = {tagset[i]:i for i in range(len(tagset))}
   # tag2index = np.random.permutation(len(tagset)) #{tagset[i]:i for i in range(len(tagset))}
   # for g in trains_graphs:
   #     g.node_features = torch.zeros(len(g.node_tags), len(tagset))
   #     g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        assert len(tagset) > 100
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        #tag2index = {tagset[i]:i for i in range(len(tagset))}
        tag2index = np.random.permutation(len(tagset)) #{tagset[i]:i for i in range(len(tagset))}
        for g in batch_graph:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            #for g in train_graphs:
            tags = set(g.node_tags)
            tag2index = np.random.permutation(len(tags))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        #print (train_graphs[selected_idx[0]].g)
        #exit()
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(F.softmax(model([graphs[j] for j in sampled_idx]), dim=-1).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()
    
    global tagset
    #tag2index = np.random.permutation(len(tagset)) #{tagset[i]:i for i in range(len(tagset))}
    #for g in train_graphs:
        #g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        #tags = set(g.node_tags)
        #tag2index = np.random.permutation(len(tags))
        #g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
    #tag2index = np.random.permutation(len(tagset)) #{tagset[i]:i for i in range(len(tagset))}

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    
    outputs = [] 
    #kkk = 1
   # if epoch > 330:
    kkk = 10
    for t in range(kkk):    
        for g in test_graphs:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            tags = set(g.node_tags)
            tag2index = np.random.permutation(len(tags))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        output = pass_data_iteratively(model, test_graphs)
        if t == 0:
            predd = output.max(1, keepdim=True)[1]
            labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
            correct = predd.eq(labels.view_as(predd)).sum().cpu().item()
            acc_test = correct / float(len(test_graphs))
            global FALLAcc
            if acc_test > FALLAcc:
                FALLAcc = acc_test
        outputs.append(output)
    output = sum(outputs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    global ALLAcc
    if acc_test > ALLAcc:
        ALLAcc = acc_test
    print("accuracy train: %f test: %f" % (acc_train, acc_test))
    print ("Best: %f %f" % (FALLAcc, ALLAcc))
    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
    
    #tagset = set([])
    global tagset
    for g in train_graphs:
        tagset = tagset.union(set(g.node_tags))
    for g in test_graphs:
        tagset = tagset.union(set(g.node_tags))
    tagset = list(tagset)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    for epoch in range(1, args.epochs):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
    #    acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

        print(model.eps)
    acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)
    

if __name__ == '__main__':
    main()
