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
        
        self.rnn = nn.LSTM(input_size, hidden_neurons, num_layers, batch_first=True,
                           dropout = 0.5)
        self.classifier = nn.Linear(hidden_neurons,input_size)
        self.sf = torch.nn.Softmax(dim=1)
        
        

    def forward(self, x):
        hidden = self.reset_h(x.size(0))
        c = self.reset_h(x.size(0))
        hidden = to_cuda(hidden)
        c = to_cuda(c)
        output, hidden = self.rnn(x,(hidden,c))
        output = output.contiguous().view(-1,self.hidden_neurons)
        output = self.classifier(output)
        #output = self.sf(output)
        
        return output, hidden
    def reset_h(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size,self.hidden_neurons)
        return hidden




class Trainer:

    def __init__(self,
                 learning_rate: float,
                 batch_size: int,
                 seq_length: int,
                 epochs: int,
                 data,
                 model: nn.Module,
                 outfile = "outfile",
                 temp: float = 0.6,
                 ):
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
        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0001)
        #self.optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate, momentum=0.7)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                            mode='min',factor=0.5, patience=2, verbose = True, threshold=1e-3)    
        
        
        self.early_stop_count = 8
        self.temp = temp
        
        #Logging
        self.outfile = "output/" + outfile
        self.checkpoint_dir = pathlib.Path("checkpoints")
        # Load our dataset
        self.data = data
        chars = list(set(self.data))
        self.data_size, self.vocab_size = len(self.data), len(chars)
        print( 'Data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
        self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(chars) }
        
        self.data = [self.char_to_ix[ch] for ch in self.data]

        self.seq_length = seq_length # number of steps to unroll the RNN for

        # Validate our model everytime we pass through 50% of the dataset
        self.global_step = 0
        
        
        # Tracking variables
        self.TRAIN_LOSS = collections.OrderedDict()
        self.VALIDATION_LOSS = collections.OrderedDict()

    def predict(self, character):
    # One-hot encoding our input to fit into the model
        seed_len = len(character)
        X = torch.zeros(1,len(character),self.vocab_size)
        #character = np.array([self.char_to_ix[c] for c in character])
        
        for j in range(seed_len):
            X[0,j,character[j]] = 1
        with torch.no_grad():
            X = to_cuda(X)
            out, hidden = self.model(X)
            sf = nn.Softmax(dim=1)
            prob = sf(out/self.temp)[-1].cpu()
        
        prob /= sum(prob)
        # Taking the class with the highest probability score from the output
        char_ind = np.random.choice(self.vocab_size,1, p=prob.numpy())
        #char_ind = torch.max(prob, dim=0)[1].item()
        return char_ind.item(), hidden



    def sample(self, out_len, start):
        self.model.eval() # eval mode
        size = out_len - len(start)
        # Now pass in the previous characters and get a new one
        for ii in range(size):
            char, h= self.predict(start)
            #chars.append(char)
            start.append(char)
        self.model.train()
        return ''.join([self.ix_to_char[ch] for ch in start])


    def should_early_stop(self):
            """
                Checks if validation loss doesn't improve over early_stop_count epochs.
            """
            # Check if we have more than early_stop_count elements in our validation_loss list.
            if len(self.VALIDATION_LOSS) < self.early_stop_count:
                return False
            # We only care about the last [early_stop_count] losses.
            relevant_loss = list(self.VALIDATION_LOSS.values())[-self.early_stop_count:]
            first_loss = relevant_loss[0]
            if first_loss == min(relevant_loss):
                print("Early stop criteria met")
                return True
            return False

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0
        
        def validation():
            with torch.no_grad():
                self.model.eval()
                X = torch.zeros(128, self.seq_length,self.vocab_size)
                Y = torch.zeros(128, self.seq_length, dtype = torch.long)
                for i in range(64):
                    inputs = self.data[i*self.seq_length:i*self.seq_length+self.seq_length]
                    targets = self.data[i*self.seq_length + 1:i*self.seq_length+self.seq_length + 1]
                    Y[i] = torch.Tensor(targets)
                    for j in range(self.seq_length):
                        X[i,j,inputs[j]] = 1
                #Compute the cross entropy loss for the batch
                X = to_cuda(X)
                Y = to_cuda(Y)
                output, hidden = self.model(X)
                print(output.size())
                val_loss = self.loss_criterion(output, Y.view(-1))
                self.model.train()
                self.VALIDATION_LOSS[self.global_step] = val_loss.detach().cpu()
            print(f"Global step: {self.global_step}, Loss: {loss}, Val loss: {val_loss}")
        
        
        def log_sample():
            length = 300
            if(self.global_step % 2000 == 0):
                length = 2000
                
            with torch.no_grad():
                seed = self.data[p:p+self.seq_length]
                seed = self.sample(length,seed)
                f = open(self.outfile, 'a')
                f.write(f"\nStep: {self.global_step}, Loss: {self.TRAIN_LOSS[self.global_step]}\n\n")
                f.write(seed)
                f.write("\n"*4 + "-"*15)
                f.close()
                print(seed)
            self.model.train()
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            p = 0
            while(not p+self.seq_length*self.batch_size+1 >= len(self.data)):
                
                loss = 0
                X = torch.zeros(self.batch_size, self.seq_length,self.vocab_size)
                Y = torch.zeros(self.batch_size, self.seq_length, dtype = torch.long)
                for i in range(self.batch_size):
                    inputs = self.data[p:p+self.seq_length]
                    targets = self.data[p+1: p + self.seq_length + 1]
                    Y[i] = torch.Tensor(targets)
                    for j in range(self.seq_length):
                        X[i,j,inputs[j]] = 1
                    p += self.seq_length
                    # Compute the cross entropy loss for the batch
                
                #print(self.data[p:p+seq_length], self.data[p+1:p + seq_length+1])
                #Y = torch.Tensor(np.array(targets))
                #print(X.size(), Y.size())
                X = to_cuda(X)
                Y = to_cuda(Y)
                output, hidden = self.model(X)
                #print(output.size())
                loss = self.loss_criterion(output, Y.view(-1))
                                    # Backpropagation
                loss.backward()
                # Gradient descent step
                self.optimizer.step()
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                
                self.TRAIN_LOSS[self.global_step] = loss.detach().cpu()
                
                if(self.global_step%300 == 0):
                    validation()
                    self.save_model()
                    self.scheduler.step(self.VALIDATION_LOSS[self.global_step])
                elif(self.global_step % 100 == 0): 
                    print(f"Global step: {self.global_step}, Loss: {loss}")
                    if(self.should_early_stop()):
                        self.save_model()
                        return
                    self.scheduler.step(self.TRAIN_LOSS[self.global_step])
                if(self.global_step % 500 == 0):
                    log_sample()
                p += self.seq_length
                self.global_step += 1
                
    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")
        utils.save_checkpoint(state_dict, filepath, is_best_model())
        
    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.TRAIN_LOSS, label="Training loss")
    utils.plot_loss(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    fName = "input/Kanye.txt"
    data = open(fName, 'r').read().lower()# should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    epochs = 100
    learning_rate = 1e-2
    seq_length = 50
    batch_size = 32
    model = MyModel(vocab_size,128, 2)
    trainer = Trainer(
        learning_rate,
        batch_size,
        seq_length,
        epochs,
        data,
        model,
        "Kanye19_03",
        0.6,
    )
    trainer.load_best_model()
    
    trainer.train()
    seed = [trainer.char_to_ix[ch] for ch in "[verse 1]"]
    trainer.sample(2000, seed)
    print("".join([trainer.ix_to_char[i] for i in seed]))
    create_plots(trainer, "task2")
