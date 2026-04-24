

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------------------------
# DATA
# -----------------------------------------------------

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822,0.4465),
        (0.2023,0.1994,0.2010)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822,0.4465),
        (0.2023,0.1994,0.2010)
    )
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

print("Train:", len(trainset), "Test:", len(testset))

# -----------------------------------------------------
# PRUNABLE LAYER
# -----------------------------------------------------

class PrunableLinear(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.bias = nn.Parameter(torch.zeros(out_features))

        # negative init → gates start nearly closed
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), -3.0) +
            torch.randn(out_features, in_features)*0.1
        )

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x, temperature=1.0):

        gates = torch.sigmoid(self.gate_scores / temperature)

        pruned_weights = self.weight * gates

        return F.linear(x, pruned_weights, self.bias), gates


# -----------------------------------------------------
# NETWORK
# -----------------------------------------------------

class PruningNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = PrunableLinear(3072,512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = PrunableLinear(512,256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = PrunableLinear(256,10)

        self.dropout = nn.Dropout(0.3)

    def forward(self,x,temperature=1.0):

        x = x.view(x.size(0),-1)

        x,g1 = self.fc1(x,temperature)
        x = self.dropout(F.relu(self.bn1(x)))

        x,g2 = self.fc2(x,temperature)
        x = self.dropout(F.relu(self.bn2(x)))

        x,g3 = self.fc3(x,temperature)

        return x,[g1,g2,g3]


# -----------------------------------------------------
# SPARSITY LOSS
# -----------------------------------------------------

def sparsity_loss(gates):

    l1 = 0

    for g in gates:

        l1 += torch.sum(g)

    return l1


# -----------------------------------------------------
# ACCURACY
# -----------------------------------------------------

def evaluate(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y in testloader:

            x,y = x.to(device),y.to(device)

            logits,_ = model(x)

            pred = logits.argmax(1)

            total += y.size(0)
            correct += (pred==y).sum().item()

    return 100*correct/total


# -----------------------------------------------------
# SPARSITY
# -----------------------------------------------------

def get_sparsity(model,threshold=0.01):

    total = 0
    pruned = 0

    with torch.no_grad():

        for m in model.modules():

            if isinstance(m,PrunableLinear):

                gates = torch.sigmoid(m.gate_scores)

                total += gates.numel()
                pruned += (gates<threshold).sum().item()

    return 100*pruned/total


# -----------------------------------------------------
# PARAM COUNT
# -----------------------------------------------------

def count_active_params(model,threshold=0.01):

    total=0
    active=0

    for m in model.modules():

        if isinstance(m,PrunableLinear):

            gates=torch.sigmoid(m.gate_scores)

            total+=gates.numel()
            active+=(gates>=threshold).sum().item()

    return active,total


# -----------------------------------------------------
# FLOPS
# -----------------------------------------------------

def compute_flops(model,threshold=0.01):

    orig=0
    pruned=0

    for m in model.modules():

        if isinstance(m,PrunableLinear):

            out_f,in_f=m.weight.shape

            orig += 2*in_f*out_f

            gates=torch.sigmoid(m.gate_scores)

            active=(gates>=threshold).sum().item()

            pruned += 2*active

    return orig,pruned


# -----------------------------------------------------
# TRAIN
# -----------------------------------------------------

def train_model(lambd,epochs=25):

    print("\nTraining λ =",lambd)

    model=PruningNet().to(device)

    criterion=nn.CrossEntropyLoss()

    gate_params=[p for n,p in model.named_parameters() if "gate_scores" in n]
    weight_params=[p for n,p in model.named_parameters() if "gate_scores" not in n]

    optimizer=optim.Adam([
        {"params":weight_params,"lr":1e-3,"weight_decay":1e-4},
        {"params":gate_params,"lr":5e-3}
    ])

    temperature=5.0

    history={"acc":[],"sparsity":[]}

    best_acc=0

    for epoch in range(epochs):

        model.train()

        total_loss=0

        for x,y in trainloader:

            x,y=x.to(device),y.to(device)

            optimizer.zero_grad()

            logits,gates=model(x,temperature)

            ce=criterion(logits,y)

            sp=sparsity_loss(gates)

            loss=ce + lambd*sp

            loss.backward()

            optimizer.step()

            total_loss+=loss.item()

        temperature*=0.92

        acc=evaluate(model)

        sparsity=get_sparsity(model)

        history["acc"].append(acc)
        history["sparsity"].append(sparsity)

        if acc>best_acc:

            best_acc=acc
            torch.save(model.state_dict(),"best_model.pth")

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss {total_loss/len(trainloader):.3f} | "
              f"Acc {acc:.2f}% | "
              f"Sparsity {sparsity:.2f}%")

    model.load_state_dict(torch.load("best_model.pth"))

    return model,history


# -----------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------

lambdas=[1e-6,1e-5,5e-5]

results=[]
models=[]
histories=[]

for lam in lambdas:

    model,history=train_model(lam)

    acc=evaluate(model)

    sparsity=get_sparsity(model)

    results.append((lam,acc,sparsity))

    models.append(model)
    histories.append(history)

    active,total=count_active_params(model)

    orig,pruned=compute_flops(model)

    print("\nCompression Report")

    print("Active Params:",active,"/",total)
    print("Pruned %:",100*(1-active/total))

    print("Original FLOPs:",orig)
    print("Pruned FLOPs:",pruned)
    print("FLOPs Reduction:",100*(1-pruned/orig))


# -----------------------------------------------------
# RESULTS TABLE
# -----------------------------------------------------

print("\nFinal Results")

print("Lambda | Accuracy | Sparsity")

for lam,acc,sp in results:

    print(lam,acc,sp)


# -----------------------------------------------------
# PLOTS
# -----------------------------------------------------

os.makedirs("plots",exist_ok=True)

# sparsity vs lambda

lam_vals=[r[0] for r in results]
sp_vals=[r[2] for r in results]
acc_vals=[r[1] for r in results]

plt.figure()

plt.plot(lam_vals,sp_vals,"o-",label="Sparsity")
plt.plot(lam_vals,acc_vals,"s--",label="Accuracy")

plt.legend()

plt.xlabel("Lambda")
plt.title("Accuracy vs Sparsity Tradeoff")

plt.savefig("plots/sparsity_vs_lambda.png")

plt.show()


# sparsity growth

plt.figure()

for i,h in enumerate(histories):

    plt.plot(h["sparsity"],label=f"λ={lambdas[i]}")

plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Sparsity")

plt.title("Sparsity Growth During Training")

plt.savefig("plots/sparsity_growth.png")

plt.show()


# gate distribution

all_gates=[]

for m in models[-1].modules():

    if isinstance(m,PrunableLinear):

        g=torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()

        all_gates.extend(g)

plt.figure(figsize=(10,5))

plt.hist(all_gates,bins=100)

plt.axvline(0.01,color="red")

plt.title("Gate Distribution")

plt.savefig("plots/gate_distribution.png")

plt.show()

print("\nAll done.")
