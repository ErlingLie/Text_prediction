import pathlib
import torch
from torch import nn
from matplotlib import pyplot as plt
import collections
import utils
import numpy as np
import random

def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements


class MyModel(nn.Module):

    def __init__(self,input_size, hidden_neurons, num_layers):

        super().__init__()
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self.output_size = input_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_neurons, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_neurons,input_size)
        #self.sf = torch.nn.Softmax(dim=1)
        
        

    def forward(self, x):
        hidden = self.reset_h(x.size(0))
        #h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_neurons) # 2 for bidirection 
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.neurons)
        #h0 = to_cuda(h0)
        #c0 = to_cua(c0)
        output, hidden = self.rnn(x,hidden)
        output = output.contiguous().view(-1,self.hidden_neurons)
        output = self.classifier(output)
        #output = self.sf(output)
        
        return output
    def reset_h(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size,self.hidden_neurons)
        return to_cuda(hidden)




class Trainer:

    def __init__(self,
                 data,
                 learning_rate: float,
                 batch_size: int,
                 seq_length: int,
                 epochs: int,
                 model: nn.Module,):
        """
            Initialize our trainer class.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)
        print(self.model)
        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)

        # Load our dataset
        self.data = data# should be simple plain text file
        chars = list(set(self.data))
        self.data_size, self.vocab_size = len(self.data), len(chars)
        print( 'data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
        self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(chars) }


        self.seq_length = seq_length # number of steps to unroll the RNN for

        # Validate our model everytime we pass through 50% of the dataset
        self.global_step = 0
        
        
        # Tracking variables
        self.TRAIN_LOSS = collections.OrderedDict()

    def predict(self, character):
    # One-hot encoding our input to fit into the model
        seed_len = len(character)
        X = torch.zeros(1,len(character),self.vocab_size)
        character = np.array([self.char_to_ix[c] for c in character])
        
        for j in range(seed_len):
            X[0,j,character[j]] = 1
        
        X = to_cuda(X)
        out = self.model(X)
        sf = nn.Softmax(dim=1)
        prob = sf(out)[-1].cpu()
        
        prob /= sum(prob)
        # Taking the class with the highest probability score from the output
        char_ind = np.random.choice(self.vocab_size,1, p=prob.numpy())
        #char_ind = torch.max(prob, dim=0)[1].item()
        return self.ix_to_char[char_ind.item()]



    def sample(self, out_len, start='hey'):
        self.model.eval() # eval mode
        start = start.lower()
        # First off, run through the starting characters
        chars = [ch for ch in start]
        size = out_len - len(chars)
        # Now pass in the previous characters and get a new one
        for ii in range(size):
            char= self.predict(chars)
            chars.append(char)
    
        return ''.join(chars)

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            p = 0
            torch.cuda.empty_cache()
            while(not p+seq_length*self.batch_size+1 >= len(self.data)):
                loss = 0
                X = torch.zeros(self.batch_size, self.seq_length,self.vocab_size)
                Y = torch.zeros(self.batch_size, self.seq_length, dtype = torch.long)
                for i in range(self.batch_size):
                    inputs = [self.char_to_ix[ch] for ch in self.data[p:p+seq_length]]
                    targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+seq_length+1]]
                    Y[i] = torch.Tensor(targets)
                    for j in range(self.seq_length):
                        X[i,j,inputs[j]] = 1
                    p += self.seq_length
                    # Compute the cross entropy loss for the batch
                X = to_cuda(X)
                Y = to_cuda(Y)
                #print(self.data[p:p+seq_length], self.data[p+1:p + seq_length+1])
                #Y = torch.Tensor(np.array(targets))
                #print(X.size(), Y.size())
                output = self.model(X)
                #print(output.size())
                loss = self.loss_criterion(output, Y.view(-1))
                                    # Backpropagation
                loss.backward(retain_graph=True)
                # Gradient descent step
                self.optimizer.step()
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                
                self.TRAIN_LOSS[self.global_step] = loss
                
                if(self.global_step%100 == 0):
                    print(f"Global step: {self.global_step}, Loss: {loss}")
                    


                if(self.global_step % 500 == 0):
                    with torch.no_grad():
                        seed = self.data[p:p+seq_length]
                        seed = self.sample(250,seed)
                        #print(output)
                        print(seed)
                    self.model.train()
                p += seq_length
                self.global_step += 1



def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.TRAIN_LOSS, label="Training loss")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    data = open("BIBLE.txt", 'r').read()# should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    epochs = 50
    learning_rate = 1e-2
    seq_length = 100
    batch_size = 8
    model = MyModel(vocab_size,100,2)
    trainer = Trainer(
        data,
        learning_rate,
        batch_size,
        seq_length,
        epochs,
        model
    )
    trainer.train()
    create_plots(trainer, "task2")
