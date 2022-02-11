import torch
import torch.nn as nn
import math

class FullyConnectedSparseLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, coef):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights).to('cuda:0')  # nn.Parameter is a Tensor that's a module parameter.
        inp_section = int(size_in/coef)
        out_section = int(size_out/coef)
        weight_canceler = [[0 for i in range(size_in)] for j in range(size_out)]     
        count = 0
        for i in range(size_out):
            for j in range(count*inp_section,count*inp_section+inp_section):
              weight_canceler[i][j] = 1
            if (i+1)%out_section==0:
              count = count + 1
 
 
        self.weight_canceler = torch.Tensor(weight_canceler).to('cuda:0')
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias).to('cuda:0')
 
        # initialize weights and biases
        self.weight_init()
 
    def weight_init(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        weights = self.weights * self.weight_canceler
        w_times_x= torch.mm(x, weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b
 
 
class SparseLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, steps):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        try:
          if int(steps):
            steps = [steps]
        except:
          pass
        weight_canceler = [[0 for i in range(size_in)] for j in range(size_out)]  
        count = 0
        
        if len(steps) ==  2:
          inp_el_counts = int(size_in/steps[0])
          out_el_counts = int(size_out/steps[1])
          steps_ratio = int(steps[0]/steps[1])
          for i in range(steps[1]):
              for j in range(i*out_el_counts,(i+1)*out_el_counts):
                  for k in range((i+1)*steps_ratio):
                      weight_canceler[j][k*inp_el_counts + count]=1
                  count = count + 1
                  if count >= inp_el_counts:
                      count = 0
        else:     
          inp_el_counts = int(size_in/steps[0])   
          for i in range(0,size_out):
              for j in range(steps[0]):
                  weight_canceler[i][j*inp_el_counts + count]=1
              count = count + 1
              if count >= inp_el_counts:
                  count = 0
 
 
        self.weight_canceler = torch.Tensor(weight_canceler).to('cuda:0')
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias).to('cuda:0')
        # initialize weights and biases
        self.weight_init()

    def weight_init(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        weights = self.weights * self.weight_canceler
        w_times_x= torch.mm(x, weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b
 
 
class TreeConnect(nn.Module):
    def __init__(self, input_dim, hidden_layers_dim , output_dim, div_coefs):
        # percentage_masked, **kwargs
        super(TreeConnect, self).__init__()
        self.output_dim = output_dim
        self.div_coefs = div_coefs 
        self.hidden_layers_dim = hidden_layers_dim
        self.first_Layer = FullyConnectedSparseLayer(input_dim, hidden_layers_dim[0],div_coefs[0]).to('cuda:0')
        #self.first_Layer = nn.Linear(input_dim, hidden_layers_dim[0])
        hidden_layers = []
        if len(hidden_layers_dim)>1:
            for i in range(1,len(hidden_layers_dim)):
                hidden_layers.append(SparseLayer(hidden_layers_dim[i-1], hidden_layers_dim[i], [div_coefs[i-1], div_coefs[i]]).to('cuda:0'))
        self.hidden_layers = hidden_layers   
        self.relu = nn.ReLU()
        self.last_layer = SparseLayer(hidden_layers_dim[-1], output_dim, div_coefs[-1]).to('cuda:0')
        
    def weight_init(self):
        self.first_Layer.weight_init()
        for hd in self.hidden_layers:
          hd.weight_init()
        self.last_layer.weight_init()


    def forward(self, x):
        x = self.first_Layer(x)
        x = self.relu(x)
        if len(self.hidden_layers)>0:
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
                x = self.relu(x)
        x = self.last_layer(x)
 
        return x
        