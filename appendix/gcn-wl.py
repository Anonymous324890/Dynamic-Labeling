import torch
import numpy as np
from itertools import permutations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("method", choices=["dynamic", "same"])
args = parser.parse_args()

GRAPH1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
GRAPH1 = list(zip(*GRAPH1))

GRAPH2 = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)]
GRAPH2 = list(zip(*GRAPH2))

NODES = 6
EMBEDDINGDIM = 8
LAYERS = 2

graph1_adj = np.zeros(shape=(NODES, NODES), dtype=np.int)
graph1_adj[GRAPH1[0], GRAPH1[1]] = 1
graph1_adj[GRAPH1[1], GRAPH1[0]] = 1

graph2_adj = np.zeros(shape=(NODES, NODES), dtype=np.int)
graph2_adj[GRAPH2[0], GRAPH2[1]] = 1
graph2_adj[GRAPH2[1], GRAPH2[0]] = 1


if args.method == "dynamic":
    inputs = np.array(list(permutations(list(range(NODES)))), dtype=np.int)
else:
    inputs = np.zeros(shape=(720, 6), dtype=np.int)

# nodes: batch, nodes, embeddingsize
# adj: nodes, nodes


class GCN(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.msg = torch.nn.Linear(embedding, embedding)
        self.combine1 = torch.nn.Linear(embedding * 2, embedding)
        self.combine2 = torch.nn.Linear(embedding, embedding)

    def forward(self, input, adj):
        msg = self.msg(input)
        msg = torch.matmul(torch.unsqueeze(adj, 0), msg)
        output = self.combine1(torch.cat([input, msg], dim=-1))
        output = torch.nn.functional.relu(output)
        output = self.combine2(output)
        return output


class GraphModel(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()

        self.embedding = torch.nn.Embedding(NODES, embedding)
        self.gcns = torch.nn.ModuleList([GCN(embedding) for _ in range(LAYERS)])
        self.output1 = torch.nn.Linear(embedding, embedding)
        self.output2 = torch.nn.Linear(embedding, 2)

    def forward(self, graph, labels):
        x = self.embedding(labels)
        for gcn in self.gcns:
            x = gcn(x, graph)
            x = torch.nn.functional.relu(x)
        x = self.output1(x)
        x = torch.nn.functional.relu(x)
        x = self.output2(x)

        x = torch.mean(x, dim=1)
        return x


model = GraphModel(EMBEDDINGDIM)
optim = torch.optim.Adam(model.parameters())

graph1_adj = torch.FloatTensor(graph1_adj)
graph2_adj = torch.FloatTensor(graph2_adj)
inputs = torch.LongTensor(inputs)

if True:
    model = model.cuda()
    graph1_adj = graph1_adj.cuda()
    graph2_adj = graph2_adj.cuda()
    inputs = inputs.cuda()

for step in range(1000):

    optim.zero_grad()

    pred1 = model(graph1_adj, inputs)
    loss1 = -torch.mean(torch.nn.functional.log_softmax(pred1, dim=-1)[:, 0], dim=0)
    pred2 = model(graph2_adj, inputs)
    loss2 = -torch.mean(torch.nn.functional.log_softmax(pred2, dim=-1)[:, 1], dim=0)
    loss = loss1 + loss2

    loss.backward()
    optim.step()

    with torch.no_grad():
        acc1 = torch.mean((torch.argmax(pred1, 1) == 0).float(), dim=0)
        acc2 = torch.mean((torch.argmax(pred2, 1) == 1).float(), dim=0)
        acc = acc1 / 2 + acc2 / 2

    print(f"step: {step:02d}, acc: {acc.item()*100: 5.02f}%, loss: {loss.item():08.06f}")


with torch.no_grad():
    pred1 = model(graph1_adj, inputs)
    acc1 = torch.mean((torch.argmax(pred1, 1) == 0).float(), dim=0)
    pred2 = model(graph2_adj, inputs)
    acc2 = torch.mean((torch.argmax(pred2, 1) == 1).float(), dim=0)
    print("Percision:", (acc1 / 2 + acc2 / 2).item())

    print("Preciction for graph 1:", pred1[0].cpu().numpy())
    print("Preciction for graph 2:", pred2[0].cpu().numpy())
