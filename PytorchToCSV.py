import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        return self.layers(x)

model = Model()

list_of_params = list(model.parameters())
# Access the weights of the first layer (for example)
print(list_of_params)

activations = 1

with open('net.csv', 'w', newline='') as csvfile:
    #writer = csv.writer(csvfile)

    l = 0
    for i in range(0, len(list_of_params), 2):
        print("Layer " + str(l + 1) + ":")
        csvfile.write("Layer," + str(l + 1) + "\n")

        for x in range(len(list_of_params[i][0])):
            print("    Neuron " + str(x + 1) + ":")
            csvfile.write("Neuron," + str(x + 1) + "\n")

            for z in range(len(list_of_params[i])):
                print("        Weight " + str(z + 1) + ": " + str(list_of_params[i][z][x]))
                csvfile.write("Weight," + str(z + 1) + "," + str(list_of_params[i][z][x].item()) + "\n")
            
            if i == 0:
                print("        Bias " + str(x + 1) + ": " + "0")
                csvfile.write("Bias," + str(x + 1) + ",0" + "\n")
            else:
                print("        Bias " + str(x + 1) + ": " + str(list_of_params[i-1][x]))
                csvfile.write("Bias," + str(x + 1) + "," + str(list_of_params[i-1][x].item()) + "\n")

        csvfile.write("\n")

        if activations > 0:
            l += 1
            csvfile.write("Layer," + str(l + 1) + "\n")
            activations -= 1

        l += 1