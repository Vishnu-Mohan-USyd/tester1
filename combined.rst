

.. include:: ./../CONTRIBUTING.rst


================
Examples
================

Samples of code snippets demonstrating usage of various modules and functions of snnTorch can be found here.

More detail is available in the `tutorials. <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_


.. # toctree directive commented out
    :maxdepth: 2
    :glob:

    examples/examples_snn
    examples/examples_splt
    examples/examples_svision
    examples/examples_sur

.. include:: ./../HISTORY.rst


snnTorch Documentation
^^^^^^^^^^^^^^^^^^^^^^

.. include:: ./../README.rst


.. # toctree directive commented out
   :maxdepth: 1
   :caption: Contents:

   readme
   installation
   snntorch
   snntorch.export
   snntorch.functional
   snntorch.spikegen
   snntorch.spikeplot
   snntorch.spikevision
   snntorch.surrogate
   snntorch.utils
   quickstart
   examples
   tutorials/index
   contributing
   history



.. # highlight directive commented out

============
Installation
============


Stable release
^^^^^^^^^^^^^^

To install snntorch, run this command in your terminal:

.. code-block:: console

    $ pip install snntorch

This is the preferred method to install snntorch, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

To install snntorch with conda:

.. code-block:: console

    $ conda install -c conda-forge snntorch

From sources
^^^^^^^^^^^^

The sources for snntorch can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/jeshraghian/snntorch

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/jeshraghian/snntorch/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/jeshraghian/snntorch
.. _tarball: https://github.com/jeshraghian/snntorch/tarball/master


=============
Quickstart 
=============

Tutorial written by Jason K. Eshraghian (`www.jasoneshraghian.com <https://www.jasoneshraghian.com>`_)


.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/quickstart.ipynb


For a comprehensive overview on how SNNs work, and what is going on
under the hood, `then you might be interested in the snnTorch tutorial
series available
here. <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__
The snnTorch tutorial series is based on the following paper. If you
find these resources or code useful in your work, please consider citing
the following source:

   `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor
   Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D.
   Lu. “Training Spiking Neural Networks Using Lessons From Deep
   Learning”. arXiv preprint arXiv:2109.12894, September
   2021. <https://arxiv.org/abs/2109.12894>`__


.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/quickstart.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


::

    pip install snntorch

::

    import torch, torch.nn as nn
    import snntorch as snn

DataLoading
^^^^^^^^^^^

Define variables for dataloading.

::

    batch_size = 128
    data_path='/tmp/data/mnist'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

Load MNIST dataset.

::

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

Define Network with snnTorch.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``snn.Leaky()`` instantiates a simple leaky integrate-and-fire
   neuron.
-  ``spike_grad`` optionally defines the surrogate gradient. If left
   undefined, the relevant gradient term is simply set to the output
   spike itself (1/0) by default.

By default, each LIF neuron returns two values: the spike and hidden state. 
But neurons chained together in ``nn.Sequential`` expect only one value. 
To handle this:

-  ``init_hidden`` initializes the hidden states (e.g., membrane
   potential) as instance variables to be processed in the background.

The final layer is not bound by this constraint, and can return multiple
tensors: 

- ``output=True`` enables the final layer to return the hidden state in addition to the spike.

::

    from snntorch import surrogate
    
    beta = 0.9  # neuron decay rate 
    spike_grad = surrogate.fast_sigmoid() # fast sigmoid surrogate gradient
    
    #  Initialize Convolutional SNN
    net = nn.Sequential(nn.Conv2d(1, 8, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(8, 16, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(16*4*4, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)

Refer to the snnTorch documentation to see more `neuron
types <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__ and
`surrogate gradient
options <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html>`__.

Define the Forward Pass
^^^^^^^^^^^^^^^^^^^^^^^

Now define the forward pass over multiple time steps of simulation.

::

    from snntorch import utils 
    
    def forward_pass(net, data, num_steps):  
      spk_rec = [] # record spikes over time
      utils.reset(net)  # reset/initialize hidden states for all LIF neurons in net
    
      for step in range(num_steps): # loop over time
          spk_out, mem_out = net(data) # one time step of the forward-pass
          spk_rec.append(spk_out) # record spikes
      
      return torch.stack(spk_rec)

Define the optimizer and loss function. Here, we use the MSE Count Loss,
which counts up the total number of output spikes at the end of the
simulation run. The correct class has a target firing rate of 80% of all
time steps, and incorrect classes are set to 20%.

::

    import snntorch.functional as SF
    
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

Objective functions do not have to be applied to the spike count. They
may be applied to the membrane potential (hidden state), or to
spike-timing targets instead of rate-based methods. A non-exhaustive
list of objective functions available include:

**Apply the objective directly to spikes:** 

* MSE Spike Count Loss: ``mse_count_loss()`` 
* Cross Entropy Spike Count Loss: ``ce_count_loss()`` 
* Cross Entropy Spike Rate Loss: ``ce_rate_loss()``

**Apply the objective to the hidden state:** 

* Cross Entropy Maximum Membrane Potential Loss: ``ce_max_membrane_loss()`` 
* MSE Membrane Potential Loss: ``mse_membrane_loss()``

For alternative objective functions, refer to the
``snntorch.functional`` `documentation
here. <https://snntorch.readthedocs.io/en/latest/snntorch.functional.html>`__

Training Loop
^^^^^^^^^^^^^

Now for the training loop. The predicted class will be set to the neuron
with the highest firing rate, i.e., a rate-coded output. We will just
measure accuracy on the training set. This training loop follows the
same syntax as with PyTorch.

::

    num_epochs = 1 # run for 1 epoch - each data sample is seen only once
    num_steps = 25  # run for 25 time steps 
    
    loss_hist = [] # record loss over iterations 
    acc_hist = [] # record accuracy over iterations
    
    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
    
            net.train() 
            spk_rec = forward_pass(net, data, num_steps) # forward-pass
            loss_val = loss_fn(spk_rec, targets) # loss calculation
            optimizer.zero_grad() # null gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights
            loss_hist.append(loss_val.item()) # store loss
    
            # print every 25 iterations
            if i % 25 == 0:
              print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
    
              # check accuracy on a single batch
              acc = SF.accuracy_rate(spk_rec, targets)  
              acc_hist.append(acc)
              print(f"Accuracy: {acc * 100:.2f}%\n")
            
            # uncomment for faster termination
            # if i == 150:
            #     break
    

More control over your model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are simulating more complex architectures, such as residual nets,
then your best bet is to wrap the network up in a class as shown below.
This time, we will explicitly use the membrane potential, ``mem``, and
let ``init_hidden`` default to false.

For the sake of speed, we’ll just simulate a fully-connected SNN, but
this can be generalized to other network types (e.g., Convs).

In addition, let’s set the neuron decay rate, ``beta``, to be a
learnable parameter. The first layer will have a shared decay rate
across neurons. Each neuron in the second layer will have an independent
decay rate. The decay is clipped between [0,1].

::

    import torch.nn.functional as F
    
    # Define Network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
    
            num_inputs = 784 # number of inputs
            num_hidden = 300 # number of hidden neurons 
            num_outputs = 10 # number of classes (i.e., output neurons)

            beta1 = 0.9 # global decay rate for all leaky neurons in layer 1
            beta2 = torch.rand((num_outputs), dtype = torch.float) # independent decay rate for each leaky neuron in layer 2: [0, 1)

            # Initialize layers
            self.fc1 = nn.Linear(num_inputs, num_hidden)
            self.lif1 = snn.Leaky(beta=beta1) # not a learnable decay rate
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            self.lif2 = snn.Leaky(beta=beta2, learn_beta=True) # learnable decay rate

        def forward(self, x):
            mem1 = self.lif1.init_leaky() # reset/init hidden states at t=0
            mem2 = self.lif2.init_leaky() # reset/init hidden states at t=0
            spk2_rec = [] # record output spikes
            mem2_rec = [] # record output hidden states

            for step in range(num_steps): # loop over time
                cur1 = self.fc1(x.flatten(1))
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)

                spk2_rec.append(spk2) # record spikes
                mem2_rec.append(mem2) # record membrane

            return torch.stack(spk2_rec), torch.stack(mem2_rec)
    
    # Load the network onto CUDA if available
    net = Net().to(device)

::

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    
    num_epochs = 1 # run for 1 epoch - each data sample is seen only once
    num_steps = 25  # run for 25 time steps 

    loss_hist = [] # record loss over iterations 
    acc_hist = [] # record accuracy over iterations
    
    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
    
            net.train() 
            spk_rec, _ = net(data) # forward-pass
            loss_val = loss_fn(spk_rec, targets) # loss calculation
            optimizer.zero_grad() # null gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights
            loss_hist.append(loss_val.item()) # store loss
    
            # print every 25 iterations
            if i % 25 == 0:
              net.eval()
              print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
    
              # check accuracy on a single batch
              acc = SF.accuracy_rate(spk_rec, targets)  
              acc_hist.append(acc)
              print(f"Accuracy: {acc * 100:.2f}%\n")
            
            # uncomment for faster termination
            # if i == 150:
            #     break
    

::

    print(f"Trained decay rate of the first layer: {net.lif1.beta:.3f}\n")
    
    print(f"Trained decay rates of the second layer: {net.lif2.beta}")

::

    # function to measure accuracy on full test set
    def test_accuracy(data_loader, net, num_steps):
      with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
    
        data_loader = iter(data_loader)
        for data, targets in data_loader:
          data = data.to(device)
          targets = targets.to(device)
          spk_rec, _ = net(data)
    
          acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
          total += spk_rec.size(1)
    
      return acc/total

::

    print(f"Test set accuracy: {test_accuracy(test_loader, net, num_steps)*100:.3f}%")

Conclusion
^^^^^^^^^^

That’s it for the quick intro to snnTorch!

-  For a detailed tutorial of spiking neurons, neural nets, encoding,
   and training using neuromorphic datasets, check out the `snnTorch
   tutorial
   series <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__.
-  For more information on the features of snnTorch, check out the
   `documentation at this
   link <https://snntorch.readthedocs.io/en/latest/>`__.
-  If you have ideas, suggestions or would like to find ways to get
   involved, then `check out the snnTorch GitHub project
   here. <https://github.com/jeshraghian/snntorch>`__


.. include:: ./../README.rst


===========================
snn.Alpha
===========================


.. automodule:: snntorch._neurons.alpha
   :members:
   :undoc-members:
   :show-inheritance:

===========================
snn.Lapicque
===========================


.. automodule:: snntorch._neurons.lapicque
   :members:
   :undoc-members:
   :show-inheritance:

===========================
snn.Leaky
===========================


.. automodule:: snntorch._neurons.leaky
   :members:
   :undoc-members:
   :show-inheritance:

===========================
snn.LeakyParallel
===========================


.. automodule:: snntorch._neurons.leakyparallel
   :members:
   :undoc-members:
   :show-inheritance:

===========================
snn.RLeaky
===========================


.. automodule:: snntorch._neurons.rleaky
   :members:
   :undoc-members:
   :show-inheritance:

===========================
snn.RSynaptic
===========================


.. automodule:: snntorch._neurons.rsynaptic
   :members:
   :undoc-members:
   :show-inheritance:

===========================
snn.SConv2dLSTM
===========================


.. automodule:: snntorch._neurons.sconv2dlstm
   :members:
   :undoc-members:
   :show-inheritance:

===========================
snn.SLSTM
===========================


.. automodule:: snntorch._neurons.slstm
   :members:
   :undoc-members:
   :show-inheritance:

===========================
snn.Synaptic
===========================


.. automodule:: snntorch._neurons.synaptic
   :members:
   :undoc-members:
   :show-inheritance:

snntorch.backprop
^^^^^^^^^^^^^^^^^
:mod:`snntorch.backprop` is a module implementing various time-variant backpropagation algorithms. Each method will perform the forward-pass, backward-pass, and parameter update across all time steps in a single line of code. 


How to use backprop
^^^^^^^^^^^^^^^^^^^
To use :mod:`snntorch.backprop` you must first construct a network, determine a loss criterion, and select an optimizer. When initializing neurons, set ``init_hidden=True``. This enables the methods in :mod:`snntorch.backprop` to automatically clear the hidden state variables, as well as detach them from the computational graph when necessary.

.. note:: The first dimension of input ``data`` is assumed to be time. The built-in backprop functions iterate through the first dimension of ``data`` by default. For time-invariant inputs, set ``time_var=False``.

Example::

      net = Net().to(device)
      optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
      criterion = nn.CrossEntropyLoss()

      # Time-variant input data 
      for input, target in dataset:
         loss = BPTT(net, input, target, num_steps, batch_size, optimizer, criterion)
      
      # Time-invariant input data
      for input, targets in dataset:
         loss = BPTT(net, input, target, num_steps, batch_size, optimizer, criterion, time_var=False)

.. automodule:: snntorch.backprop
   :members:
   :undoc-members:
   :show-inheritance:

snntorch.export
^^^^^^^^^^^^^^^
:mod:`snntorch.export` is a module that enables cross-compatibility with other SNN libraries by converting snntorch models to a `Neuromorphic Intermediate Representation (NIR) <https://nnir.readthedocs.io/en/latest/>`_

.. automodule:: snntorch.export
   :members:
   :undoc-members:
   :show-inheritance:

snntorch.functional
^^^^^^^^^^^^^^^^^^^
:mod:`snntorch.functional` implements common arithmetic operations applied to spiking neurons, such as loss and regularization functions, and state quantization etc.


How to use functional
^^^^^^^^^^^^^^^^^^^^^
To use :mod:`snntorch.functional` you assign the function state to a variable, and then call that variable.

Example::

      import snntorch as snn
      import snntorch.functional as SF

      net = Net().to(device)
      optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
      criterion = SF.ce_count_loss()  # apply cross-entropy to spike count

      spk_rec, mem_rec = net(input_data)
      loss = loss_fn(spk_rec, targets)

      optimizer.zero_grad()
      loss.backward()

      # Weight Update
      optimizer.step()


Accuracy Functions
^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.acc
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.loss
   :members:
   :undoc-members:
   :show-inheritance:

Regularization Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.reg
   :members:
   :undoc-members:
   :show-inheritance:

State Quantization
^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.quant
   :members:
   :undoc-members:
   :show-inheritance:
   
Probe
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.functional.probe
   :members:
   :undoc-members:
   :show-inheritance:


snntorch
^^^^^^^^


snnTorch Neurons
^^^^^^^^^^^^^^^^
:mod:`snntorch` is designed to be intuitively used with PyTorch, as though each spiking neuron were simply another activation in a sequence of layers. 

A variety of spiking neuron classes are available which can simply be treated as activation units with PyTorch. 
Each layer of spiking neurons are therefore agnostic to fully-connected layers, convolutional layers, residual connections, etc. 

The neuron models are represented by recursive functions which removes the need to store membrane potential traces in order to calculate the gradient. 
The lean requirements of :mod:`snntorch` enable small and large networks to be viably trained on CPU, where needed. 
Being deeply integrated with ``torch.autograd``, :mod:`snntorch` is able to take advantage of GPU acceleration in the same way as PyTorch.

By default, PyTorch's autodifferentiation mechanism in ``torch.autograd`` nulls the gradient signal of the spiking neuron graph due to non-differentiable spiking threshold functions.
:mod:`snntorch` overrides the default gradient by using :mod:`snntorch.neurons.Heaviside`. Alternative options exist in :mod:`snntorch.surrogate`.

At present, the neurons available in :mod:`snntorch` are variants of the Leaky Integrate-and-Fire neuron model:

* **Leaky** - 1st-Order Leaky Integrate-and-Fire Neuron
* **RLeaky** - As above, with recurrent connections for output spikes
* **Synaptic** - 2nd-Order Integrate-and-Fire Neuron (including synaptic conductance)
* **RSynaptic** - As above, with recurrent connections for output spikes
* **Lapicque** - Lapicque's RC Neuron Model
* **Alpha** - Alpha Membrane Model

Neuron models that accelerate training require passing data in parallel. Available neurons include:
* **LeakyParallel** - 1st Order Leaky Integrate-and-Fire Neuron

Additional models include spiking-LSTMs and spiking-ConvLSTMs:

* **SLSTM** - Spiking long short-term memory cell with state-thresholding 
* **SConv2dLSTM** - Spiking 2d convolutional short-term memory cell with state thresholding



How to use snnTorch's neuron models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following arguments are common across most neuron models:

* **threshold** - firing threshold of the neuron
* **spike_grad** - surrogate gradient function (see :mod:`snntorch.surrogate`)
* **init_hidden** - setting to ``True`` hides all neuron states as instance variables to reduce code complexity
* **inhibition** - setting to ``True`` enables only the neuron with the highest membrane potential to fire in a dense layer (not for use in convs etc.)
* **learn_beta** - setting to ``True`` enables the decay rate to be a learnable parameter
* **learn_threshold** - setting to ``True`` enables the threshold to be a learnable parameter 
* **reset_mechanism** - options include ``subtract`` (reset-by-subtraction), ``zero`` (reset-to-zero), and ``none`` (no reset mechanism: i.e., leaky integrator neuron)
* **output** - if ``init_hidden=True``, the spiking neuron will only return the output spikes. Setting ``output=True`` enables the hidden state(s) to be returned as well. Useful when using ``torch.nn.sequential``. 

Leaky integrate-and-fire neuron models also include:

* **beta** - decay rate of membrane potential, clipped between 0 and 1 during the forward-pass. Can be a single-value tensor (same decay for all neurons in a layer), or can be multi-valued (individual weights p/neuron in a layer. More complex neurons include additional parameters, such as **alpha**.

Recurrent spiking neuron models, such as :mod:`snntorch.RLeaky` and :mod:`snntorch.RSynaptic` explicitly pass the output spike back to the input. 
Such neurons include additional arguments:

* **V** - Recurrent weight. Can be a single-valued tensor (same weight across all neurons in a layer), or multi-valued tensor (individual weights p/neuron in a layer).
* **learn_V** - defaults to ``True``, which enables **V** to be a learnable parameter. 

Spiking neural networks can be constructed using a combination of the ``snntorch`` and ``torch.nn`` packages.

Example::

      import torch
      import torch.nn as nn
      import snntorch as snn

      alpha = 0.9
      beta = 0.85

      num_steps = 100


      # Define Network
      class Net(nn.Module):
         def __init__(self):
            super().__init__()

            # initialize layers
            self.fc1 = nn.Linear(num_inputs, num_hidden)
            self.lif1 = snn.Leaky(beta=beta)
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            self.lif2 = snn.Leaky(beta=beta)

         def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()

            spk2_rec = []  # Record the output trace of spikes
            mem2_rec = []  # Record the output trace of membrane potential

            for step in range(num_steps):
                  cur1 = self.fc1(x.flatten(1))
                  spk1, mem1 = self.lif1(cur1, mem1)
                  cur2 = self.fc2(spk1)
                  spk2, mem2 = self.lif2(cur2, mem2)

                  spk2_rec.append(spk2)
                  mem2_rec.append(mem2)

            return torch.stack(spk2_rec), torch.stack(mem2_rec)

      net = Net().to(device)

      output, mem_rec = net(data)

In the above example, the hidden state ``mem`` must be manually initialized for each layer.
This can be overcome by automatically instantiating neuron hidden states by invoking ``init_hidden=True``. 

In some cases (e.g., truncated backprop through time), it might be necessary to perform backward passes before all time steps have completed processing.
This requires moving the time step for-loop out of the network and into the training-loop. 

An example of this is shown below::

      import torch
      import torch.nn as nn
      import snntorch as snn

      num_steps = 100

      lif1 = snn.Leaky(beta=0.9, init_hidden=True) # only returns spk
      lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True) # returns mem and spk if output=True


      #  Initialize Network
      net = nn.Sequential(nn.Flatten(),
                          nn.Linear(784,1000), 
                          lif1,
                          nn.Linear(1000, 10),
                          lif2).to(device)

      for step in range(num_steps):
         spk_out, mem_out = net(data)


Setting the hidden states to instance variables is necessary for calling :mod:`nn.Sequential` from PyTorch.

Whenever a neuron is instantiated, it is added as a list item to the class variable :mod:`LIF.instances`. 
This allows you to keep track of what neurons are being used in the network, and to detach neurons from the computation graph. 

In the above examples, the decay rate of membrane potential :mod:`beta` is treated as a hyperparameter. 
But it can also be configured as a learnable parameter, as shown below::

      import torch
      import torch.nn as nn
      import snntorch as snn

      num_steps = 100

      lif1 = snn.Leaky(beta=0.9, learn_beta=True, init_hidden=True) # only returns spk
      lif2 = snn.Leaky(beta=0.5, learn_beta=True, init_hidden=True, output=True) # returns mem and spk if output=True


      #  Initialize Network
      net = nn.Sequential(nn.Flatten(),
                          nn.Linear(784,1000), 
                          lif1,
                          nn.Linear(1000, 10),
                          lif2).to(device)

      for step in range(num_steps):
         spk_out, mem_out = net(data.view(batch_size, -1))

Here, :mod:`beta` is initialized to 0.9 for the first layer, and 0.5 for the second layer.
Each layer then treats it as a learnable parameter, just like all the other network weights.
In the event you wish to have a learnable decay rate for each neuron rather than each layer, the following example shows how::


      import torch
      import torch.nn as nn
      import snntorch as snn

      num_steps = 100
      num_hidden = 1000
      num_output = 10

      beta1 = torch.rand(num_hidden)  # randomly initialize beta as a vector
      beta2 = torch.rand(num_output)

      lif1 = snn.Leaky(beta=beta1, learn_beta=True, init_hidden=True) # only returns spk
      lif2 = snn.Leaky(beta=beta2 learn_beta=True, init_hidden=True, output=True) # returns mem and spk if output=True

      #  Initialize Network
      net = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, num_hidden), 
                          lif1,
                          nn.Linear(1000, num_output),
                          lif2).to(device)

      for step in range(num_steps):
         spk_out, mem_out = net(data.view(batch_size, -1))

The same approach as above can be used for implementing learnable thresholds, using ``learn_threshold=True``.

Each neuron has the option to inhibit other neurons within the same dense layer from firing. 
This can be invoked by setting ``inhibition=True`` when instantiating the neuron layer. It has not yet been implemented for networks other than fully-connected layers, so use with caution.

Neuron List
^^^^^^^^^^^

.. # toctree directive commented out
    :maxdepth: 2
    :titlesonly:
    :glob:

    snn.neurons_*

snnTorch Layers
^^^^^^^^^^^^^^^

.. automodule:: snntorch._layers.bntt
   :members:
   :undoc-members:
   :show-inheritance:


Neuron Parent Classes
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch._neurons.neurons 
   :members:
   :undoc-members:
   :show-inheritance:

snntorch.spikegen
^^^^^^^^^^^^^^^^^
:mod:`snntorch.spikegen` is a module that provides a variety of common spike generation and conversion methods, including spike-rate and latency coding.

How to use spikegen
^^^^^^^^^^^^^^^^^^^
In general, tensors containing non-spiking data can simply be passed into one of the functions in :mod:`snntorch.spikegen` to convert them into discrete spikes.
There are a variety of methods to achieve this conversion. At present, `snntorch` supports:

* `rate coding`_
* `latency coding`_
* `delta modulation`_

.. _rate coding: https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.rate
.. _latency coding: https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.latency
.. _delta modulation: https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.delta

There are also options for converting targets into time-varying spikes.

.. automodule:: snntorch.spikegen
   :members:
   :undoc-members:
   :show-inheritance:

snntorch.spikeplot
^^^^^^^^^^^^^^^^^^
:mod:`snntorch.spikeplot` is deeply integrated with `matplotlib.pyplot` and `celluloid`.
It serves to reduce the amount of boilerplate code required  to generate a variety of animations and plots.

.. automodule:: snntorch.spikeplot
   :members:
   :undoc-members:
   :show-inheritance:


snntorch.spikevision
^^^^^^^^^^^^^^^^^^^^

.. warning::
    The spikevision module has been deprecated.
    To load neuromorphic datasets, we recommend using the `Tonic project <https://github.com/neuromorphs/tonic>`_.
    For examples on how to use snnTorch together with Tonic, please refer to `Tutorial 7 in the snnTorch Tutorial Series <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html>`_.


The :code:`spikevision` module consists of neuromorphic datasets and common image transformations.
It is the neuromorphic analog to `torchvision <https://pytorch.org/vision/stable/index.html>`_. 


:code:`spikevision` contains the following neuromorphic datasets:

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Dataset
     - Description
     - Author URL
   * - `NMNIST <https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.spikedata.html#nmnist>`_
     - | A spiking version of the original 
       | frame-based `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.
     - `G. Orchard <https://www.garrickorchard.com/datasets/n-mnist>`_
   * - `DVSGesture <https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.spikedata.html#dvsgesture>`_
     - | 11 hand gestures recorded from 29 subjects 
       | under 3 illumination conditions using a DVS128.
     - `IBM Research <https://www.research.ibm.com/dvsgesture/>`_
   * - `SHD <https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.spikedata.html#shd>`_
     - | Spikes in 700 input channels were generated 
       | using an artificial cochlea model listening 
       | to studio recordings of spoken digits from 
       | 0 to 9 in both German and English languages.
     - `Zenke Lab <https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/>`_


**Module Reference:**

.. # toctree directive commented out
    :maxdepth: 2
    :glob:

    snntorch.spikevision.spikedata


snntorch.spikevision.spikedata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All datasets are subclasses of :code:`torch.utils.data.Dataset` i.e., they have :code:`__getitem__` and :code:`__len__` methods implemented. 
Hence, they can all be passed to a :code:`torch.utils.data.DataLoader` which can load multiple samples in parallel using :code:`torch.multiprocessing` workers. 
For example::

   nmnist_data = spikevision.data.NMNIST('path/to/nmnist_root/')
   data_loader = DataLoader(nmnist_data, 
                            batch_size=4,
                            shuffle=True, 
                            num_workers=args.nThreads)


For further examples on each dataset and its use, please refer to the `examples <https://snntorch.readthedocs.io/en/latest/examples/examples_svision.html>`_.

NMNIST
^^^^^^

.. autoclass:: snntorch.spikevision.spikedata.nmnist.NMNIST
   :members:


DVSGesture
^^^^^^^^^^

.. autoclass:: snntorch.spikevision.spikedata.dvs_gesture.DVSGesture
   :members:



SHD
^^^

.. autoclass:: snntorch.spikevision.spikedata.shd.SHD
   :members:



snntorch.surrogate
^^^^^^^^^^^^^^^^^^

By default, PyTorch's autodifferentiation tools are unable to calculate the analytical derivative of the spiking neuron graph. 
The discrete nature of spikes makes it difficult for ``torch.autograd`` to calculate a gradient that facilitates learning.
:mod:`snntorch` overrides the default gradient by using :mod:`snntorch.surrogate.ATan`.

Alternative gradients are also available in the :mod:`snntorch.surrogate` module. 
These represent either approximations of the backward pass or probabilistic models of firing as a function of the membrane potential.
Custom, user-defined surrogate gradients can also be implemented.

At present, the surrogate gradient functions available include:

* `Sigmoid <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html#snntorch.surrogate.Sigmoid>`_
* `Fast Sigmoid <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html#snntorch.surrogate.FastSigmoid>`_
* `ATan <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html#snntorch.surrogate.ATan>`_
* `Straight Through Estimator <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html#snntorch.surrogate.StraightThroughEstimator>`_
* `Triangular <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html#snntorch.surrogate.Triangular>`_
* `SpikeRateEscape <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html#snntorch.surrogate.SpikeRateEscape>`_
* `Custom Surrogate Gradients <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html#snntorch.surrogate.CustomSurrogate>`_

amongst several other options. 

For further reading, see:

    *E. O. Neftci, H. Mostafa, F. Zenke (2019) Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks. IEEE Signal Processing Magazine, pp. 51-63.*

How to use surrogate
^^^^^^^^^^^^^^^^^^^^

The surrogate gradient must be passed as the ``spike_grad`` argument to the neuron model. 
If ``spike_grad`` is left unspecified, it defaults to :mod:`snntorch.surrogate.ATan`.
In the following example, we apply the fast sigmoid surrogate to :mod:`snntorch.Synaptic`.

Example::

   import snntorch as snn
   from snntorch import surrogate
   import torch
   import torch.nn as nn

   alpha = 0.9
   beta = 0.85

   # Initialize surrogate gradient
   spike_grad1 = surrogate.fast_sigmoid()  # passes default parameters from a closure
   spike_grad2 = surrogate.FastSigmoid.apply  # passes default parameters, equivalent to above
   spike_grad3 = surrogate.fast_sigmoid(slope=50)  # custom parameters from a closure

   # Define Network
   class Net(nn.Module):
    def __init__(self):
        super().__init__()

    # Initialize layers, specify the ``spike_grad`` argument
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad1)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad3)

    def forward(self, x, syn1, mem1, spk1, syn2, mem2):
        cur1 = self.fc1(x)
        spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
        cur2 = self.fc2(spk1)
        spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
        return syn1, mem1, spk1, syn2, mem2, spk2

    net = Net().to(device)

Custom Surrogate Gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^

For flexibility, custom surrogate gradients can also be defined by the user
using `custom_surrogate`.


Example::

   import snntorch as snn
   from snntorch import surrogate
   import torch
   import torch.nn as nn

   beta = 0.9

   # Define custom surrogate gradient
    def custom_fast_sigmoid(input_, grad_input, spikes):
        ## The hyperparameter slope is defined inside the function.
        slope = 25
        grad = grad_input / (slope * torch.abs(input_) + 1.0) ** 2
        return grad

    spike_grad = surrogate.custom_surrogate(custom_fast_sigmoid)

   # Define Network
   class Net(nn.Module):
    def __init__(self):
        super().__init__()

    # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, mem1, spk1, mem2):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        return mem1, spk1, mem2, spk2

    net = Net().to(device)


List of surrogate gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: snntorch.surrogate
   :members:
   :undoc-members:
   :show-inheritance:


snntorch.utils
^^^^^^^^^^^^^^

:mod:`snntorch.utils` contains a handful of utility functions for handling datasets.

.. automodule:: snntorch.utils
   :members:
   :undoc-members:
   :show-inheritance:


===========================
Tutorial 1 - Spike Encoding
===========================

Tutorial written by Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_

In this tutorial, you will learn how to use snnTorch to: 

  * convert datasets into spiking datasets, 
  * how to visualise them, 
  * and how to generate random spike trains.


Introduction
^^^^^^^^^^^^

Light is what we see when the retina converts photons into spikes. Odors
are what we smell when volatilised molecules are converted into spikes.
Touch is what we feel when nerve endings turn tactile pressure into
spikes. The brain trades in the global currency of the *spike*.

If our end goal is to build a spiking neural network (SNN), it makes
sense to use spikes at the input too. Although it is quite common to use
non-spiking inputs (as will be seen in Tutorial 3), part of the appeal
of encoding data come from the *three S’s*: spikes, sparsity, and static
suppression.

1. **Spikes**: (a-b) Biological neurons process and communicate via
   spikes, which are electrical impulses of approximately 100 mV in
   amplitude. (c) Many computational models of neurons simplify this
   voltage burst to a discrete, single-bit event: a ‘1’ or a ‘0’. This
   is far simpler to represent in hardware than a high precision value.

2. **Sparsity**: (c) Neurons spend most of their time at rest, silencing
   most activations to *zero* at any given time. Not only are sparse
   vectors/tensors (with loads of zeros) cheap to store, but say we need
   to multiply sparse activations with synaptic weights. If most values
   are multiplied by ‘0’, then we don’t need to read many of the network
   parameters from memory. This means neuromorphic hardware can be
   extremely efficient.

3. **Static-Suppression (a.k.a, event-driven processing**: (d-e) The
   sensory periphery only processes information when there is new
   information to process. Each pixel in (e) responds to *changes* in
   illuminance, so most of the image is blocked out. Conventional signal
   processing requires all channels/pixels to adhere to a global
   sampling/shutter rate, which slows down how frequently sensing can
   take place. Event-driven processing now only contributes to sparsity
   and power-efficiency by blocking unchanging input, but it often
   allows for much faster processing speeds.

   .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/3s.png?raw=true
            :align: center
            :width: 800


In this tutorial, we will assume we have some non-spiking input data
(i.e., the MNIST dataset) and that we want to encode it into spikes
using a few different techniques. So let’s get started!

Install the latest PyPi distribution of snnTorch:

::

    $ pip install snntorch

1. Setting up the MNIST Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.1. Import packages and setup environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    import snntorch as snn
    import torch

::

    # Training Parameters
    batch_size=128
    data_path='/tmp/data/mnist'
    num_classes = 10  # MNIST has 10 output classes
    
    # Torch Variables
    dtype = torch.float

1.2 Download Dataset
^^^^^^^^^^^^^^^^^^^^

::

    from torchvision import datasets, transforms
    
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

If the above code block throws an error, e.g. the MNIST servers are
down, then uncomment the following code instead.

::

    # # temporary dataloader if MNIST service is unavailable
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz
    
    # mnist_train = datasets.MNIST(root = './', train=True, download=True, transform=transform)

Until we actually start training a network, we won’t need large
datasets. ``snntorch.utils`` contains a few useful functions for
modifying datasets. Apply ``data_subset`` to reduce the dataset
by the factor defined in ``subset``. E.g., for ``subset=10``, a
training set of 60,000 will be reduced to 6,000.

::

    from snntorch import utils
    
    subset = 10
    mnist_train = utils.data_subset(mnist_train, subset)

::

    >>> print(f"The size of mnist_train is {len(mnist_train)}")
    The size of mnist_train is 6000


1.3 Create DataLoaders
^^^^^^^^^^^^^^^^^^^^^^

The Dataset objects created above load data into memory, and the
DataLoader will serve it up in batches. DataLoaders in PyTorch are a
handy interface for passing data into a network. They return an iterator
divided up into mini-batches of size ``batch_size``.

::

    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

2. Spike Encoding
^^^^^^^^^^^^^^^^^

Spiking Neural Networks (SNNs) are made to exploit time-varying data.
And yet, MNIST is not a time-varying dataset. There are two options for using MNIST with an SNN:

1. Repeatedly pass the same training sample
   :math:`\mathbf{X}\in\mathbb{R}^{m\times n}` to the network at each
   time step. This is like converting MNIST into a static, unchanging video.
   Each element of :math:`\mathbf{X}` can take a high
   precision value normalized between 0 and 1: :math:`X_{ij}\in [0, 1]`.
   

   .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_1_static.png?raw=true
            :align: center
            :width: 800

2. Convert the input into a spike train of sequence length
   ``num_steps``, where each feature/pixel takes on a discrete value
   :math:`X_{i,j} \in \{0, 1\}`. In this case, MNIST is converted into a time-varying sequence of spikes that features a relation to the original image.

    .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_2_spikeinput.png?raw=true
              :align: center
              :width: 800

The first method is quite straightforward, and does not fully exploit
the temporal dynamics of SNNs. So let’s consider data-to-spike conversion (encoding) from (2) in more detail.

The module ``snntorch.spikegen`` (i.e., spike generation) contains a
series of functions that simplify the conversion of data into spikes.
There are currently three options available for spike encoding in
``snntorch``:

1. Rate coding:
   `spikegen.rate <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.rate>`__
2. Latency coding:
   `spikegen.latency <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.latency>`__
3. Delta modulation:
   `spikegen.delta <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.delta>`__

How do these differ?

1. *Rate coding* uses input features to determine spiking **frequency**
2. *Latency coding* uses input features to determine spike **timing**
3. *Delta modulation* uses the temporal **change** of input features to
   generate spikes

2.1 Rate coding of MNIST
^^^^^^^^^^^^^^^^^^^^^^^^

One example of converting input data into a rate code is as follows.
Each normalised input feature :math:`X_{ij}` is used as the probability
an event (spike) occurs at any given time step, returning a rate-coded
value :math:`R_{ij}`. This can be treated as a Bernoulli trial:
:math:`R_{ij}\sim B(n,p)`, where the number of trials is :math:`n=1`,
and the probability of success (spiking) is :math:`p=X_{ij}`.
Explicitly, the probability a spike occurs is:

.. math:: {\rm P}(R_{ij}=1) = X_{ij} = 1 - {\rm P}(R_{ij} = 0)

Create a vector filled with the value ‘0.5’ and encode it using
the above technique:

::

    # Temporal Dynamics
    num_steps = 10
    
    # create vector filled with 0.5
    raw_vector = torch.ones(num_steps)*0.5
    
    # pass each sample through a Bernoulli trial
    rate_coded_vector = torch.bernoulli(raw_vector)

::
^^
    >>> print(f"Converted vector: {rate_coded_vector}")
    Converted vector: tensor([1., 1., 1., 0., 0., 1., 1., 0., 1., 0.])
    
    >>> print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")
    The output is spiking 60.00% of the time.

Now try again, but increasing the length of ``raw_vector``:

::

    num_steps = 100
    
    # create vector filled with 0.5
    raw_vector = torch.ones(num_steps)*0.5
    
    # pass each sample through a Bernoulli trial
    rate_coded_vector = torch.bernoulli(raw_vector)
    >>> print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")
    The output is spiking 48.00% of the time.
 
As ``num_steps``\ :math:`\rightarrow\infty`, the proportion of spikes
approaches the original raw value.

For an MNIST image, this probability of spiking corresponds to the pixel
value. A white pixel corresponds to a 100% probability of spiking, and a
black pixel will never generate a spike. Take a look at the ‘Rate
Coding’ column below for further intuition.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_3_spikeconv.png?raw=true
        :align: center
        :width: 1000

In a similar way, ``spikegen.rate`` can be used to generate a rate-coded
sample of data. As each sample of MNIST is just an image, we can use
``num_steps`` to repeat it across time.

::

    from snntorch import spikegen
    
    # Iterate through minibatches
    data = iter(train_loader)
    data_it, targets_it = next(data)
    
    # Spiking Data
    spike_data = spikegen.rate(data_it, num_steps=num_steps)

If the input falls outside of :math:`[0,1]`, this no longer represents a
probability. Such cases are automatically clipped to ensure the feature
represents a probability.

The structure of the input data is
``[num_steps x batch_size x input dimensions]``:

::

    >>> print(spike_data.size())
    torch.Size([100, 128, 1, 28, 28])

2.2 Visualization
^^^^^^^^^^^^^^^^^

2.2.1 Animation
^^^^^^^^^^^^^^^

snnTorch contains a module
`snntorch.spikeplot <https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html>`__
that simplifies the process of visualizing, plotting, and animating
spiking neurons.

::

    import matplotlib.pyplot as plt
    import snntorch.spikeplot as splt
    from IPython.display import HTML

To plot one sample of data, index into a single sample from the batch (B) dimension 
of ``spike_data``, ``[T x B x 1 x 28 x 28]``:

::

    spike_data_sample = spike_data[:, 0, 0]
    >>> print(spike_data_sample.size())
    torch.Size([100, 28, 28])

``spikeplot.animator`` makes it super simple to animate 2-D data. Note:
if you are running the notebook locally on your desktop, please
uncomment the line below and modify the path to your ffmpeg.exe

::

    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'
    
    HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator.mp4?raw=true"></video>
  </center>

::

    # If you're feeling sentimental, you can save the animation: .gif, .mp4 etc.
    anim.save("spike_mnist_test.mp4")

The associated target label can be indexed as follows:

::

    >>> print(f"The corresponding target is: {targets_it[0]}")
    The corresponding target is: 7

MNIST features a greyscale image, and the white text guarantees a 100%
of spiking at every time step. So let’s do that again but reduce the
spiking frequency. This can be achieved by setting the argument
``gain``. Here, we will reduce spiking frequency to 25%.

::

    spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)
    
    spike_data_sample2 = spike_data[:, 0, 0]
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample2, fig, ax)
    HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator-25.mp4?raw=true"></video>
  </center>

::

    # Uncomment for optional save
    # anim.save("spike_mnist_test2.mp4")

Now average the spikes out over time and reconstruct the input
images.

::

    plt.figure(facecolor="w")
    plt.subplot(1,2,1)
    plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 1')
    
    plt.subplot(1,2,2)
    plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 0.25')
    
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/gain.png?raw=true
        :align: center
        :width: 300

The case where ``gain=0.25`` is lighter than where ``gain=1``, as
spiking probability has been reduced by a factor of :math:`\times 4`.

2.2.2 Raster Plots
^^^^^^^^^^^^^^^^^^

Alternatively, we can generate a raster plot of an input sample. This
requires reshaping the sample into a 2-D tensor, where ‘time’ is the
first dimension. Pass this sample into the function
``spikeplot.raster``.

::

    # Reshape
    spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))
    
    # raster plot
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data_sample2, ax, s=1.5, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster.png?raw=true
        :align: center
        :width: 600

The following code snippet shows how to index into one single neuron. 
Depending on the input data, you may need to try
a few different neurons between 0 & 784 before finding one that
spikes.

::
^^
    idx = 210  # index into 210th neuron

    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    splt.raster(spike_data_sample.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")
    
    plt.title("Input Neuron")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster1.png?raw=true
        :align: center
        :width: 400

2.2.3 Summary of Rate Coding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The idea of rate coding is actually quite controversial. Although we are
fairly confident rate coding takes place at our sensory periphery, we
are not convinced that the cortex globally encodes information as spike
rates. A couple of compelling reasons why include:

-  **Power Consumption:** Nature optimised for efficiency. Multiple
   spikes are needed to achieve any sort of task, and each spike
   consumes power. In fact, `Olshausen and Field’s work in “What is the
   other 85% of V1
   doing?” <http://www.rctn.org/bruno/papers/V1-chapter.pdf>`__
   demonstrates that rate-coding can only explain, at most, the activity
   of 15% of neurons in the primary visual cortex (V1). It is unlikely
   to be the only mechanism within the brain, which is both
   resource-constrained and highly efficient.

-  **Reaction Response Times:** We know that the reaction time of a
   human is roughly around 250ms. If the average firing rate of a neuron
   in the human brain is on the order of 10Hz, then we can only process
   about 2 spikes within our reaction timescale.

So why, then, might we use rate codes if they are not optimal for power
efficiency or latency? Even if our brain doesn’t process data as a rate,
we are fairly sure that our biological sensors do. The power/latency
disadvantages are partially offset by showing huge noise robustness:
it’s fine if some of the spikes fail to generate, because there will be
plenty more where they came from.

Additionally, you may have heard of the `Hebbian mantra of “neurons that
fire together, wire together” <https://doi.org/10.2307/1418888>`__. If
there is plenty of spiking, this may suggest there is plenty of
learning. In some cases where training SNNs proves to be challenging,
encouraging more firing via a rate code is one possible solution.

Rate coding is almost certainly working in conjunction with other
encoding schemes in the brain. We will consider these other encoding
mechanisms in the following sections. This covers the ``spikegen.rate`` function. 
Further information `can be
found in the documentation
here <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`__.

2.3 Latency Coding of MNIST
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Temporal codes capture information about the precise firing time of
neurons; a single spike carries much more meaning than in rate codes
which rely on firing frequency. While this opens up more susceptibility
to noise, it can also decrease the power consumed by the hardware
running SNN algorithms by orders of magnitude.

``spikegen.latency`` is a function that allows each input to fire at
most **once** during the full time sweep. Features closer to ``1`` will
fire earlier and features closer to ``0`` will fire later. I.e., in our
MNIST case, bright pixels will fire earlier and dark pixels will fire
later.

The following block derives how this works. If you’ve forgotten circuit
theory and/or the math means nothing to you, then don’t worry! All that
matters is: **big** input means **fast** spike; **small** input means
**late** spike.

------------------------

*Optional: Derivation of Latency Code Mechanism*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, spike timing is calculated by treating the input feature as the current injection :math:`I_{in}` into an RC circuit. This current moves charge onto the capacitor, which increases :math:`V(t)`. We assume that there is a trigger voltage, :math:`V_{thr}`, which once reached, generates a spike. The question then becomes: *for a given input current (and equivalently, input feature), how long does it take for a spike to be generated?*

Starting with Kirchhoff's current law, :math:`I_{in} = I_R + I_C`, the rest of the derivation leads us to a logarithmic relationship between time and the input. 

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_4_latencyrc.png?raw=true
        :align: center
        :width: 500

------------------------

The following function uses the above result to convert a feature of intensity
:math:`X_{ij}\in [0,1]` into a latency coded response :math:`L_{ij}`.

::

    def convert_to_time(data, tau=5, threshold=0.01):
      spike_time = tau * torch.log(data / (data - threshold))
      return spike_time 

Now, use the above function to visualize the relationship between input feature intensity and its corresponding spike time.

::

    raw_input = torch.arange(0, 5, 0.05) # tensor from 0 to 5
    spike_times = convert_to_time(raw_input)
    
    plt.plot(raw_input, spike_times)
    plt.xlabel('Input Value')
    plt.ylabel('Spike Time (s)')
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/spike_time.png?raw=true
        :align: center
        :width: 400

The smaller the value, the later the spike occurs with exponential
dependence.

The vector ``spike_times`` contains the time at which spikes are triggered, rather than a sparse tensor that contains the spikes themselves (1's and 0's). 
When running an SNN simulation, we need the 1/0 representation to obtain all of the advantages of using spikes.
This whole process can be automated using ``spikegen.latency``, where we pass a minibatch from the MNIST dataset in `data_it`:

::

    spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

Some of the arguments include:

-  ``tau``: the RC time constant of the circuit. By default, the input features are treated as a constant
   current injected into an RC circuit. A higher ``tau`` will induce slower firing.
-  ``threshold``: the membrane potential firing threshold. Input values below this threshold do not have a closed-form solution, as the input current is insufficient to drive the membrane up to the threshold. All values below the threshold are clipped and assigned to the final time step.

2.3.1 Raster plot
^^^^^^^^^^^^^^^^^

::

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()
    
    # optional save
    # fig.savefig('destination_path.png', format='png', dpi=300)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster2.png?raw=true
        :align: center
        :width: 600

To make sense of the raster plot, note that high intensity
features fire first, whereas low intensity features fire last:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_5_latencyraster.png?raw=true
        :align: center
        :width: 800

The logarithmic code coupled with the lack of diverse input values
(i.e., the lack of midtone/grayscale features) causes significant
clustering in two areas of the plot. The bright pixels induce firing at
the start of the run, and the dark pixels at the end. We can increase ``tau`` to slow down the spike times, or linearize the spike times by setting the optional argument ``linear=True``.

::

    spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear=True)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster3.png?raw=true
        :align: center
        :width: 600

The spread of firing times is much more evenly distributed now. This is
achieved by linearizing the logarithmic equation according to the
rules shown below. Unlike the RC model, there is no physical basis for
the model. It’s just simpler.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_6_latencylinear.png?raw=true
        :align: center
        :width: 600

But note how all firing occurs within the first ~5 time steps, whereas the
simulation range is 100 time steps. This indicates there are many
redundant time steps doing nothing. This can be solved by either
increasing ``tau`` to slow down the time constant, or setting the
optional argument ``normalize=True`` to span the full range of
``num_steps``.

::

    spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01,
                                  normalize=True, linear=True)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster4.png?raw=true
        :align: center
        :width: 600

One major advantage of latency coding over rate coding is
sparsity. If neurons are constrained to firing a maximum of
once over the time course of interest, then this promotes low-power
operation.

In the scenario shown above, a majority of the spikes occur at the final
time step, where the input features fall below the threshold. In a
sense, the dark background of the MNIST sample holds no useful information.

We can remove these redundant features by setting ``clip=True``.

::

    spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, 
                                  clip=True, normalize=True, linear=True)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster5.png?raw=true
        :align: center
        :width: 600

That looks much better!

2.3.2 Animation
^^^^^^^^^^^^^^^

We will run the exact same code block as before to create an animation.

::

    >>> spike_data_sample = spike_data[:, 0, 0]
    >>> print(spike_data_sample.size())
    torch.Size([100, 28, 28])

::

    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)
    
    HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator2.mp4?raw=true"></video>
  </center>

This animation is obviously much tougher to make out in video form, but
a keen eye will be able to catch a glimpse of the initial frame where
most of the spikes occur. Index into the corresponding target
value to check its value.

::

    # Save output: .gif, .mp4 etc.
    # anim.save("mnist_latency.gif")

::

    >>> print(targets_it[0])
    tensor(4, device='cuda:0')


That’s it for the ``spikegen.latency`` function. Further information
`can be found in the documentation
here <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`__.

2.4 Delta Modulation
^^^^^^^^^^^^^^^^^^^^

There are theories that the retina is adaptive: it will only process
information when there is something new to process. If there is no
change in your field of view, then your photoreceptor cells are
less prone to firing.

That is to say: **biology is event-driven**. Neurons thrive on
change.

As a nifty example, a few researchers have dedicated their lives to
designing retina-inspired image sensors, for example, the `Dynamic
Vision
Sensor <https://ieeexplore.ieee.org/abstract/document/7128412/>`__.
Although `the attached link is from over a decade ago, the work in this
video <https://www.youtube.com/watch?v=6eOM15U_t1M&ab_channel=TobiDelbruck>`__
was ahead of its time.

Delta modulation is based on event-driven spiking. The
``snntorch.delta`` function accepts a time-series tensor as input. It
takes the difference between each subsequent feature across all time
steps. By default, if the difference is both *positive* and *greater
than the threshold* :math:`V_{thr}`, a spike is generated:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_7_delta.png?raw=true
        :align: center
        :width: 600

To illustrate, let’s first come up with a contrived example where we
create our own input tensor.

::

    # Create a tensor with some fake time-series data
    data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])
    
    # Plot the tensor
    plt.plot(data)
    
    plt.title("Some fake time-series data")
    plt.xlabel("Time step")
    plt.ylabel("Voltage (mV)")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/fake_data.png?raw=true
      :align: center
      :width: 300

Pass the above tensor into the ``spikegen.delta`` function, with
an arbitrarily selected ``threshold=4``:

::

    # Convert data
    spike_data = spikegen.delta(data, threshold=4)
    
    # Create fig, ax
    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    # Raster plot of delta converted data
    splt.raster(spike_data, ax, c="black")
    
    plt.title("Input Neuron")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.xlim(0, len(data))
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/delta.png?raw=true
        :align: center
        :width: 400


There are three time steps where the difference between :math:`data[T]`
and :math:`data[T+1]` is greater than or equal to :math:`V_{thr}=4`.
This means there are three *on-spikes*.

The large dip to :math:`-20` has not been captured in our spikes. It
might be that we care about negative swings as well, in which case we
can enable the optional argument ``off_spike=True``.

::

    # Convert data
    spike_data = spikegen.delta(data, threshold=4, off_spike=True)
    
    # Create fig, ax
    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    # Raster plot of delta converted data
    splt.raster(spike_data, ax, c="black")
    
    plt.title("Input Neuron")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.xlim(0, len(data))
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/delta2.png?raw=true
        :align: center
        :width: 400

We’ve generated additional spikes, but this isn’t actually the full
picture!

Printing out the tensor will show the presence of “off-spikes” which take on a value of ``-1``.

::

    >>> print(spike_data)
    tensor([ 0.,  0.,  0.,  0.,  1., -1.,  1., -1.,  1.,  0.,  0.])

While ``spikegen.delta`` has only been demonstrated on a fake sample of data, 
its true use is to compress time-series data by only generating spikes for sufficiently large changes/events. 

That wraps up the three main spike conversion functions! There are still
additional features to each of the three conversion techniques that have
not been detailed in this tutorial. In particular, we have only looked
at encoding input data; we have not considered how we might encode
targets, and when that is necessary. We recommend `referring to the
documentation for a deeper
dive <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`__.

3. Spike Generation (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now what if we don’t actually have any data to start with? Say we just
want a randomly generated spike train from scratch. Inside of
``spikegen.rate`` is a nested function, ``rate_conv``, which actually
performs the spike conversion step.

All we have to do is initialize a randomly generated ``torchTensor`` to
pass in.

::

    # Create a random spike train
    spike_prob = torch.rand((num_steps, 28, 28), dtype=dtype) * 0.5
    spike_rand = spikegen.rate_conv(spike_prob)

3.1 Animation
^^^^^^^^^^^^^

::

    fig, ax = plt.subplots()
    anim = splt.animator(spike_rand, fig, ax)
    
    HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/rand_spikes.mp4?raw=true"></video>
  </center>


::

    # Save output: .gif, .mp4 etc.
    # anim.save("random_spikes.gif")

3.2 Raster
^^^^^^^^^^

::

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_rand[:, 0].view(num_steps, -1), ax, s=25, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/rand_raster.png?raw=true
      :align: center
      :width: 600

Conclusion
^^^^^^^^^^

That’s it for spike conversion and generation. This approach generalizes
beyond images, to single-dimensional and multi-dimensional tensors.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.

For reference, the documentation for `spikegen can be found
here <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`__
and for `spikeplot,
here <https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html>`__.

`In the next
tutorial <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__, 
you will learn the basics of spiking neurons and how to use them.

Additional Resources 
^^^^^^^^^^^^^^^^^^^^^

* `Check out the snnTorch GitHub project here. <https://github.com/jeshraghian/snntorch>`__

======================================================
Tutorial 2 - The Leaky Integrate-and-Fire Neuron
======================================================

Tutorial written by Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


Introduction
^^^^^^^^^^^^

In this tutorial, you will: 

* Learn the fundamentals of the leaky integrate-and-fire (LIF) neuron model 
* Use snnTorch to implement a first order LIF neuron

Install the latest PyPi distribution of snnTorch:

::

    $ pip install snntorch

::

    # imports
    import snntorch as snn
    from snntorch import spikeplot as splt
    from snntorch import spikegen
    
    import torch
    import torch.nn as nn
    
    import numpy as np
    import matplotlib.pyplot as plt


1. The Spectrum of Neuron Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A large variety of neuron models are out there, ranging from
biophysically accurate models (i.e., the Hodgkin-Huxley models) to the
extremely simple artificial neuron that pervades all facets of modern
deep learning.

**Hodgkin-Huxley Neuron Models**\ :math:`-`\ While biophysical models
can reproduce electrophysiological results with a high degree of
accuracy, their complexity makes them difficult to use at present.

**Artificial Neuron Model**\ :math:`-`\ On the other end of the spectrum
is the artificial neuron. The inputs are multiplied by their
corresponding weights and passed through an activation function. This
simplification has enabled deep learning researchers to perform
incredible feats in computer vision, natural language processing, and
many other machine learning-domain tasks.

**Leaky Integrate-and-Fire Neuron Models**\ :math:`-`\ Somewhere in the
middle of the divide lies the leaky integrate-and-fire (LIF) neuron
model. It takes the sum of weighted inputs, much like the artificial
neuron. But rather than passing it directly to an activation function,
it will integrate the input over time with a leakage, much like an RC
circuit. If the integrated value exceeds a threshold, then the LIF
neuron will emit a voltage spike. The LIF neuron abstracts away the
shape and profile of the output spike; it is simply treated as a
discrete event. As a result, information is not stored within the spike,
but rather the timing (or frequency) of spikes. Simple spiking neuron
models have produced much insight into the neural code, memory, network
dynamics, and more recently, deep learning. The LIF neuron sits in the
sweet spot between biological plausibility and practicality.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_1_neuronmodels.png?raw=true
        :align: center
        :width: 1000

The different versions of the LIF model each have their own dynamics and
use-cases. snnTorch currently supports the following LIF neurons: 

* Lapicque’s RC model: ``snntorch.Lapicque`` 
* 1st-order model: ``snntorch.Leaky`` 
* Synaptic Conductance-based neuron model: ``snntorch.Synaptic``
* Recurrent 1st-order model: ``snntorch.RLeaky``
* Recurrent Synaptic Conductance-based neuron model: ``snntorch.RSynaptic``
* Alpha neuron model: ``snntorch.Alpha``

Several other non-LIF spiking neurons are also available. 
This tutorial focuses on the first of these models. This will
be used to build towards the other models in `subsequent tutorials <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_.

2. The Leaky Integrate-and-Fire Neuron Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.1 Spiking Neurons: Intuition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In our brains, a neuron might be connected to 1,000 :math:`-` 10,000
other neurons. If one neuron spikes, all downhill neurons might
feel it. But what determines whether a neuron spikes in the first place?
The past century of experiments demonstrate that if a neuron experiences
*sufficient* stimulus at its input, then it might become excited and fire its own spike. 

Where does this stimulus come from? It could be from:

* the sensory periphery, 
* an invasive electrode artificially stimulating the neuron, or in most cases, 
* from other pre-synaptic neurons.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_2_intuition.png?raw=true
        :align: center
        :width: 600

Given that these spikes are very short bursts of electrical activity, it
is quite unlikely for all input spikes to arrive at the neuron body in
precise unison. This indicates the presence of temporal dynamics that
‘sustain’ the input spikes, kind of like a delay.

2.2 The Passive Membrane
^^^^^^^^^^^^^^^^^^^^^^^^

Like all cells, a neuron is surrounded by a thin membrane. This membrane
is a lipid bilayer that insulates the conductive saline solution within
the neuron from the extracellular medium. Electrically, the two
conductive solutions separated by an insulator act as a capacitor.

Another function of this membrane is to control what goes in and out of
this cell (e.g., ions such as Na\ :math:`^+`). The membrane is usually
impermeable to ions which blocks them from entering and exiting the
neuron body. But there are specific channels in the membrane that are
triggered to open by injecting current into the neuron. This charge
movement is electrically modelled by a resistor.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_3_passivemembrane.png?raw=true
        :align: center
        :width: 450

The following block will derive the behaviour of a LIF neuron from
scratch. If you’d prefer to skip the math, then feel free to scroll on
by; we’ll take a more hands-on approach to understanding the LIF neuron
dynamics after the derivation.

------------------------

**Optional: Derivation of LIF Neuron Model**

Now say some arbitrary time-varying current :math:`I_{\rm in}(t)` is injected into the neuron, 
be it via electrical stimulation or from other neurons. The total current in the circuit is conserved, so:

.. math:: I_{\rm in}(t) = I_{R} + I_{C}

From Ohm's Law, the membrane potential measured between the inside 
and outside of the neuron :math:`U_{\rm mem}` is proportional to 
the current through the resistor:

.. math:: I_{R}(t) = \frac{U_{\rm mem}(t)}{R}

The capacitance is a proportionality constant between the charge 
stored on the capacitor :math:`Q` and :math:`U_{\rm mem}(t)`:

.. math:: Q = CU_{\rm mem}(t)

The rate of change of charge gives the capacitive current:

.. math:: \frac{dQ}{dt}=I_C(t) = C\frac{dU_{\rm mem}(t)}{dt}

Therefore:

.. math:: I_{\rm in}(t) = \frac{U_{\rm mem}(t)}{R} + C\frac{dU_{\rm mem}(t)}{dt}

.. math:: \implies RC \frac{dU_{\rm mem}(t)}{dt} = -U_{\rm mem}(t) + RI_{\rm in}(t)

The right hand side of the equation is of units 
**\[Voltage]**. On the left hand side of the equation, 
the term :math:`\frac{dU_{\rm mem}(t)}{dt}` is of units 
**\[Voltage/Time]**. To equate it to the left hand side (i.e., voltage), 
:math:`RC` must be of unit **\[Time]**. We refer to :math:`\tau = RC` as the time constant of the circuit:

.. math:: \tau \frac{dU_{\rm mem}(t)}{dt} = -U_{\rm mem}(t) + RI_{\rm in}(t)

The passive membrane is therefore described by a linear differential equation.

For a derivative of a function to be of the same form as the original function, 
i.e., :math:`\frac{dU_{\rm mem}(t)}{dt} \propto U_{\rm mem}(t)`, this implies 
the solution is exponential with a time constant :math:`\tau`.

Say the neuron starts at some value :math:`U_{0}` with no further input, 
i.e., :math:`I_{\rm in}(t)=0.` The solution of the linear differential equation is:

.. math:: U_{\rm mem}(t) = U_0e^{-\frac{t}{\tau}}

The general solution is shown below.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_RCmembrane.png?raw=true
        :align: center
        :width: 450

------------------------


**Optional: Forward Euler Method to Solving the LIF Neuron Model**

We managed to find the analytical solution to the LIF neuron, but it is 
unclear how this might be useful in a neural network. This time,
let’s instead use the forward Euler method to solve the previous linear
ordinary differential equation (ODE). This approach might seem
arduous, but it gives us a discrete, recurrent representation of the LIF
neuron. Once we reach this solution, it can be applied directly to a neural
network. As before, the linear ODE describing the RC circuit is:

.. math:: \tau \frac{dU(t)}{dt} = -U(t) + RI_{\rm in}(t)

The subscript from :math:`U(t)` is omitted for simplicity.

First, let’s solve this derivative without taking the limit
:math:`\Delta t \rightarrow 0`:

.. math:: \tau \frac{U(t+\Delta t)-U(t)}{\Delta t} = -U(t) + RI_{\rm in}(t)

For a small enough :math:`\Delta t`, this gives a good enough
approximation of continuous-time integration. Isolating the membrane at
the following time step gives:

.. math:: U(t+\Delta t) = U(t) + \frac{\Delta t}{\tau}\big(-U(t) + RI_{\rm in}(t)\big)

The following function represents this equation:

::

    def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
      tau = R*C
      U = U + (time_step/tau)*(-U + I*R)
      return U

The default values are set to :math:`R=50 M\Omega` and
:math:`C=100pF` (i.e., :math:`\tau=5ms`). These are quite
realistic with respect to biological neurons.

Now loop through this function, iterating one time step at a time.
The membrane potential is initialized at :math:`U=0.9 V`, with the assumption that
there is no injected input current, :math:`I_{\rm in}=0 A`.
The simulation is performed with a millisecond precision
:math:`\Delta t=1\times 10^{-3}`\ s.

::

    num_steps = 100
    U = 0.9
    U_trace = []  # keeps a record of U for plotting
    
    for step in range(num_steps):
      U_trace.append(U)
      U = leaky_integrate_neuron(U)  # solve next step of U
    
    plot_mem(U_trace, "Leaky Neuron Model")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/leaky1.png?raw=true
        :align: center
        :width: 300

This exponential decay seems to match what we expected!

3 Lapicque’s LIF Neuron Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This similarity between nerve membranes and RC circuits was observed by
`Louis Lapicque in
1907 <https://pubmed.ncbi.nlm.nih.gov/17968583/>`__. He stimulated
the nerve fiber of a frog with a brief electrical pulse, and found that neuron
membranes could be approximated as a capacitor with a leakage. We pay
homage to his findings by naming the basic LIF neuron model in snnTorch
after him.

Most of the concepts in Lapicque’s model carry forward to other LIF
neuron models. Now it's time to simulate this neuron using snnTorch.

3.1 Lapicque: Without Stimulus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instantiate Lapicque’s neuron using the following line of code.
R & C are modified to simpler values, while keeping the previous time
constant of :math:`\tau=5\times10^{-3}`\ s.

::

    time_step = 1e-3
    R = 5
    C = 1e-3
    
    # leaky integrate and fire neuron, tau=5e-3
    lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

The neuron model is now stored in ``lif1``. To use this neuron:

**Inputs** 

* ``cur_in``: each element of :math:`I_{\rm in}` is sequentially passed as an input (0 for now) 
* ``mem``: the membrane potential, previously :math:`U[t]`, is also passed as input. Initialize it arbitrarily as :math:`U[0] = 0.9~V`.

**Outputs** 

* ``spk_out``: output spike :math:`S_{\rm out}[t+\Delta t]` at the next time step (‘1’ if there is a spike; ‘0’ if there is no spike) 
* ``mem``: membrane potential :math:`U_{\rm mem}[t+\Delta t]` at the next time step

These all need to be of type ``torch.Tensor``.

::

    # Initialize membrane, input, and output
    mem = torch.ones(1) * 0.9  # U=0.9 at t=0
    cur_in = torch.zeros(num_steps, 1)  # I=0 for all t 
    spk_out = torch.zeros(1)  # initialize output spikes

These values are only for the initial time step :math:`t=0`. 
To analyze the evolution of ``mem`` over time, create a list ``mem_rec`` to record these values at every time step.

::

    # A list to store a recording of membrane potential
    mem_rec = [mem]

Now it’s time to run a simulation! At each time step, ``mem`` is
updated and stored in ``mem_rec``:

::

    # pass updated value of mem and cur_in[step]=0 at every time step
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in[step], mem)
    
      # Store recordings of membrane potential
      mem_rec.append(mem)
    
    # convert the list of tensors into one tensor
    mem_rec = torch.stack(mem_rec)
    
    # pre-defined plotting function
    plot_mem(mem_rec, "Lapicque's Neuron Model Without Stimulus")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque.png?raw=true
        :align: center
        :width: 300

The membrane potential decays over time in the absence of any input
stimuli.

3.2 Lapicque: Step Input
^^^^^^^^^^^^^^^^^^^^^^^^

Now apply a step current :math:`I_{\rm in}(t)` that switches on at
:math:`t=t_0`. Given the linear first-order differential equation:

.. math::  \tau \frac{dU_{\rm mem}}{dt} = -U_{\rm mem} + RI_{\rm in}(t),

the general solution is:

.. math:: U_{\rm mem}=I_{\rm in}(t)R + [U_0 - I_{\rm in}(t)R]e^{-\frac{t}{\tau}}

If the membrane potential is initialized to
:math:`U_{\rm mem}(t=0) = 0 V`, then:

.. math:: U_{\rm mem}(t)=I_{\rm in}(t)R [1 - e^{-\frac{t}{\tau}}]

Based on this explicit time-dependent form, we expect
:math:`U_{\rm mem}` to relax exponentially towards :math:`I_{\rm in}R`.
Let’s visualize what this looks like by triggering a current pulse of
:math:`I_{in}=100mA` at :math:`t_0 = 10ms`.

::

    # Initialize input current pulse
    cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.1), 0)  # input current turns on at t=10
    
    # Initialize membrane, output and recordings
    mem = torch.zeros(1)  # membrane potential of 0 at t=0
    spk_out = torch.zeros(1)  # neuron needs somewhere to sequentially dump its output spikes
    mem_rec = [mem]

This time, the new values of ``cur_in`` are passed to the neuron:

::

    num_steps = 200
    
    # pass updated value of mem and cur_in[step] at every time step
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in[step], mem)
      mem_rec.append(mem)
    
    # crunch -list- of tensors into one tensor
    mem_rec = torch.stack(mem_rec)
    
    plot_step_current_response(cur_in, mem_rec, 10)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_step.png?raw=true
        :align: center
        :width: 450

As :math:`t\rightarrow \infty`, the membrane potential
:math:`U_{\rm mem}` exponentially relaxes to :math:`I_{\rm in}R`:

::

    >>> print(f"The calculated value of input pulse [A] x resistance [Ω] is: {cur_in[11]*lif1.R} V")
    >>> print(f"The simulated value of steady-state membrane potential is: {mem_rec[200][0]} V")
    
    The calculated value of input pulse [A] x resistance [Ω] is: 0.5 V
    The simulated value of steady-state membrane potential is: 0.4999999403953552 V

Close enough!

3.3 Lapicque: Pulse Input
^^^^^^^^^^^^^^^^^^^^^^^^^

Now what if the step input was clipped at :math:`t=30ms`?

::

    # Initialize current pulse, membrane and outputs
    cur_in1 = torch.cat((torch.zeros(10, 1), torch.ones(20, 1)*(0.1), torch.zeros(170, 1)), 0)  # input turns on at t=10, off at t=30
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec1 = [mem]

::

    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in1[step], mem)
      mem_rec1.append(mem)
    mem_rec1 = torch.stack(mem_rec1)
    
    plot_current_pulse_response(cur_in1, mem_rec1, "Lapicque's Neuron Model With Input Pulse", 
                                vline1=10, vline2=30)


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse1.png?raw=true
        :align: center
        :width: 450

:math:`U_{\rm mem}` rises just as it did for the step input, but now it
decays with a time constant of :math:`\tau` as in our first simulation.

Let’s deliver approximately the same amount of charge
:math:`Q = I \times t` to the circuit in half the time. This means the
input current amplitude must be increased by a little, and the
time window must be decreased.

::

    # Increase amplitude of current pulse; half the time.
    cur_in2 = torch.cat((torch.zeros(10, 1), torch.ones(10, 1)*0.111, torch.zeros(180, 1)), 0)  # input turns on at t=10, off at t=20
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec2 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in2[step], mem)
      mem_rec2.append(mem)
    mem_rec2 = torch.stack(mem_rec2)
    
    plot_current_pulse_response(cur_in2, mem_rec2, "Lapicque's Neuron Model With Input Pulse: x1/2 pulse width",
                                vline1=10, vline2=20)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse2.png?raw=true
        :align: center
        :width: 450


Let’s do that again, but with an even faster input pulse and higher
amplitude:

::

    # Increase amplitude of current pulse; quarter the time.
    cur_in3 = torch.cat((torch.zeros(10, 1), torch.ones(5, 1)*0.147, torch.zeros(185, 1)), 0)  # input turns on at t=10, off at t=15
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec3 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in3[step], mem)
      mem_rec3.append(mem)
    mem_rec3 = torch.stack(mem_rec3)
    
    plot_current_pulse_response(cur_in3, mem_rec3, "Lapicque's Neuron Model With Input Pulse: x1/4 pulse width",
                                vline1=10, vline2=15)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse3.png?raw=true
        :align: center
        :width: 450


Now compare all three experiments on the same plot:


::

    compare_plots(cur_in1, cur_in2, cur_in3, mem_rec1, mem_rec2, mem_rec3, 10, 15, 
                  20, 30, "Lapicque's Neuron Model With Input Pulse: Varying inputs")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/compare_pulse.png?raw=true
        :align: center
        :width: 450

As the input current pulse amplitude increases, the rise time of the
membrane potential speeds up. In the limit of the input current pulse
width becoming infinitesimally small, :math:`T_W \rightarrow 0s`, the
membrane potential will jump straight up in virtually zero rise time:

::

    # Current spike input
    cur_in4 = torch.cat((torch.zeros(10, 1), torch.ones(1, 1)*0.5, torch.zeros(189, 1)), 0)  # input only on for 1 time step
    mem = torch.zeros(1) 
    spk_out = torch.zeros(1)
    mem_rec4 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in4[step], mem)
      mem_rec4.append(mem)
    mem_rec4 = torch.stack(mem_rec4)
    
    plot_current_pulse_response(cur_in4, mem_rec4, "Lapicque's Neuron Model With Input Spike", 
                                vline1=10, ylim_max1=0.6)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_spike.png?raw=true
        :align: center
        :width: 450


The current pulse width is now so short, it effectively looks like a
spike. That is to say, charge is delivered in an infinitely short period
of time, :math:`I_{\rm in}(t) = Q/t_0` where :math:`t_0 \rightarrow 0`.
More formally:

.. math:: I_{\rm in}(t) = Q \delta (t-t_0),

where :math:`\delta (t-t_0)` is the Dirac-Delta function. Physically, it
is impossible to ‘instantaneously’ deposit charge. But integrating
:math:`I_{\rm in}` gives a result that makes physical sense, as we can
obtain the charge delivered:

.. math:: 1 = \int^{t_0 + a}_{t_0 - a}\delta(t-t_0)dt

.. math:: f(t_0) = \int^{t_0 + a}_{t_0 - a}f(t)\delta(t-t_0)dt

Here,
:math:`f(t_0) = I_{\rm in}(t_0=10) = 0.5A \implies f(t) = Q = 0.5C`.

Hopefully you have a good feel of how the membrane potential leaks at
rest, and integrates the input current. That covers the ‘leaky’ and
‘integrate’ part of the neuron. How about the fire?

3.4 Lapicque: Firing
^^^^^^^^^^^^^^^^^^^^

So far, we have only seen how a neuron will react to spikes at the
input. For a neuron to generate and emit its own spikes at the output,
the passive membrane model must be combined with a threshold.

If the membrane potential exceeds this threshold, then a voltage spike
will be generated, external to the passive membrane model.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_spiking.png?raw=true
        :align: center
        :width: 400

Modify the ``leaky_integrate_neuron`` function from before to add
a spike response.

::

    # R=5.1, C=5e-3 for illustrative purposes
    def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
      tau_mem = R*C
      spk = (mem > threshold) # if membrane exceeds threshold, spk=1, else, 0
      mem = mem + (time_step/tau_mem)*(-mem + cur*R)
      return mem, spk

Set ``threshold=1``, and apply a step current to get this neuron
spiking.

::

    # Small step current input
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron simulation
    for step in range(num_steps):
      mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="LIF Neuron Model With Uncontrolled Spiking")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lif_uncontrolled.png?raw=true
        :align: center
        :width: 450


Oops - the output spikes have gone out of control! This is
because we forgot to add a reset mechanism. In reality, each time a
neuron fires, the membrane potential hyperpolarizes back to its resting
potential.

Implementing this reset mechanism into our neuron:

::

    # LIF w/Reset mechanism
    def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
      tau_mem = R*C
      spk = (mem > threshold)
      mem = mem + (time_step/tau_mem)*(-mem + cur*R) - spk*threshold  # every time spk=1, subtract the threhsold
      return mem, spk

::

    # Small step current input
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron simulation
    for step in range(num_steps):
      mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="LIF Neuron Model With Reset")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/reset_2.png?raw=true
        :align: center
        :width: 450

Bam. We now have a functional leaky integrate-and-fire neuron model!

Note that if :math:`I_{\rm in}=0.2 A` and :math:`R<5 \Omega`, then
:math:`I\times R < 1 V`. If ``threshold=1``, then no spiking would
occur. Feel free to go back up, change the values, and test it out.

As before, all of that code is condensed by calling the built-in Lapicque neuron model from snnTorch:

::

    # Create the same neuron as before using snnTorch
    lif2 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)
    
    >>> print(f"Membrane potential time constant: {lif2.R * lif2.C:.3f}s")
    "Membrane potential time constant: 0.025s"

::

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.2), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Simulation run across 100 time steps.
    for step in range(num_steps):
      spk_out, mem = lif2(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Step Input")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_reset.png?raw=true
        :align: center
        :width: 450

The membrane potential exponentially rises and then hits the threshold,
at which point it resets. We can roughly see this occurs between
:math:`105ms < t_{\rm spk} < 115ms`. As a matter of curiousity, let’s
see what the spike recording actually consists of:

::

    >>> print(spk_rec[105:115].view(-1))
    tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

The absence of a spike is represented by :math:`S_{\rm out}=0`, and the
occurrence of a spike is :math:`S_{\rm out}=1`. Here, the spike occurs
at :math:`S_{\rm out}[t=109]=1`. If you are wondering why each of these entries is stored as a tensor, it
is because in future tutorials we will simulate large scale neural
networks. Each entry will contain the spike responses of many neurons,
and tensors can be loaded into GPU memory to speed up the training
process.

If :math:`I_{\rm in}` is increased, then the membrane potential
approaches the threshold :math:`U_{\rm thr}` faster:

::

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.3), 0)  # increased current
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif2(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
^^^^
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Periodic Firing")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/periodic.png?raw=true
        :align: center
        :width: 450

A similar increase in firing frequency can also be induced by decreasing
the threshold. This requires initializing a new neuron model, but the
rest of the code block is the exact same as above:

::

    # neuron with halved threshold
    lif3 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5)
    
    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.3), 0) 
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif3(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=0.5, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Lower Threshold")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/threshold.png?raw=true
        :align: center
        :width: 450

That’s what happens for a constant current injection. But in both deep
neural networks and in the biological brain, most neurons will be
connected to other neurons. They are more likely to receive spikes,
rather than injections of constant current.

3.5 Lapicque: Spike Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let’s harness some of the skills we learnt in `Tutorial
1 <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb>`__,
and use the ``snntorch.spikegen`` module to create some randomly
generated input spikes.

::

    # Create a 1-D random spike train. Each element has a probability of 40% of firing.
    spk_in = spikegen.rate_conv(torch.ones((num_steps,1)) * 0.40)

Run the following code block to see how many spikes have been generated.

::

    >>> print(f"There are {int(sum(spk_in))} total spikes out of {len(spk_in)} time steps.")
    There are 85 total spikes out of 200 time steps.

::

    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    splt.raster(spk_in.reshape(num_steps, -1), ax, s=100, c="black", marker="|")
    plt.title("Input Spikes")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/spikes.png?raw=true
        :align: center
        :width: 400

::

    # Initialize inputs and outputs
    mem = torch.ones(1)*0.5
    spk_out = torch.zeros(1)
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif3(spk_in[step], mem)
      spk_rec.append(spk_out)
      mem_rec.append(mem)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_spk_mem_spk(spk_in, mem_rec, spk_out, "Lapicque's Neuron Model With Input Spikes")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/spk_mem_spk.png?raw=true
        :align: center
        :width: 450

3.6 Lapicque: Reset Mechanisms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We already implemented a reset mechanism from scratch, but let’s dive a
little deeper. This sharp drop of membrane potential promotes a
reduction of spike generation, which supplements part of the theory on
how brains are so power efficient. Biologically, this drop of membrane
potential is known as ‘hyperpolarization’. Following that, it is
momentarily more difficult to elicit another spike from the neuron.
Here, we use a reset mechanism to model hyperpolarization.

There are two ways to implement the reset mechanism:

1. *reset by subtraction* (default) :math:`-` subtract the threshold
   from the membrane potential each time a spike is generated;
2. *reset to zero* :math:`-` force the membrane potential to zero each
   time a spike is generated.
3. *no reset* :math:`-` do nothing, and let the firing go potentially uncontrolled.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_5_reset.png?raw=true
        :align: center
        :width: 400

Instantiate another neuron model to demonstrate how to alternate
between reset mechanisms. By default, snnTorch neuron models use ``reset_mechanism = "subtract"``.
This can be explicitly overridden by passing the argument
``reset_mechanism =  "zero"``.

::

    # Neuron with reset_mechanism set to "zero"
    lif4 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5, reset_mechanism="zero")
    
    # Initialize inputs and outputs
    spk_in = spikegen.rate_conv(torch.ones((num_steps, 1)) * 0.40)
    mem = torch.ones(1)*0.5
    spk_out = torch.zeros(1)
    mem_rec0 = [mem]
    spk_rec0 = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif4(spk_in[step], mem)
      spk_rec0.append(spk_out)
      mem_rec0.append(mem)
    
    # convert lists to tensors
    mem_rec0 = torch.stack(mem_rec0)
    spk_rec0 = torch.stack(spk_rec0)
    
    plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0)


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/comparison.png?raw=true
        :align: center
        :width: 550

Pay close attention to the evolution of the membrane potential,
especially in the moments after it reaches the threshold. You may notice
that for “Reset to Zero”, the membrane potential is forced back to zero
after each spike.

So which one is better? Applying ``"subtract"`` (the default value in
``reset_mechanism``) is less lossy, because it does not ignore how much
the membrane exceeds the threshold by.

On the other hand, applying a hard reset with ``"zero"`` promotes
sparsity and potentially less power consumption when running on
dedicated neuromorphic hardware. Both options are available for you to
experiment with.

That covers the basics of a LIF neuron model!

Conclusion
^^^^^^^^^^

In practice, we probably wouldn’t use this neuron model to train a
neural network. The Lapicque LIF model has added a lot of
hyperparameters to tune: :math:`R`, :math:`C`, :math:`\Delta t`,
:math:`U_{\rm thr}`, and the choice of reset mechanism. It’s all a
little bit daunting. So the `next tutorial <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ will eliminate most of these
hyperparameters, and introduce a neuron model that is better suited for
large-scale deep learning.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.

For reference, the documentation `can be found
here <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__.

Further Reading
^^^^^^^^^^^^^^^

-  `Check out the snnTorch GitHub project here. <https://github.com/jeshraghian/snntorch>`__
-  `snnTorch
   documentation <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__
   of the Lapicque, Leaky, Synaptic, and Alpha models
-  `Neuronal Dynamics: From single neurons to networks and models of
   cognition <https://neuronaldynamics.epfl.ch/index.html>`__ by Wulfram
   Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.
-  `Theoretical Neuroscience: Computational and Mathematical Modeling of
   Neural
   Systems <https://mitpress.mit.edu/books/theoretical-neuroscience>`__
   by Laurence F. Abbott and Peter Dayan


================================================
Tutorial 3 - A Feedforward Spiking Neural Network
================================================

Tutorial written by Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:
   
    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


Introduction
^^^^^^^^^^^^

In this tutorial, you will: 

* Learn how to simplify the leaky integrate-and-fire (LIF) neuron to make it deep learning-friendly 
* Implement a feedforward spiking neural network (SNN)

Install the latest PyPi distribution of snnTorch:

::

    $ pip install snntorch

::

    # imports
    import snntorch as snn
    from snntorch import spikeplot as splt
    from snntorch import spikegen
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt


1. Simplifying the Leaky Integrate-and-Fire Neuron Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous tutorial, we designed our own LIF neuron model. But it was quite complex, and added an array of
hyperparameters to tune, including :math:`R`, :math:`C`,
:math:`\Delta t`, :math:`U_{\rm thr}`, and the choice of reset
mechanism. This is a lot to keep track of, and only grows more cumbersome
when scaled up to full-blown SNN. So let’s make a few
simplfications.

1.1 The Decay Rate: beta
^^^^^^^^^^^^^^^^^^^^^^^^

In the previous tutorial, the Euler method was used to derive the
following solution to the passive membrane model:

.. math:: U(t+\Delta t) = (1-\frac{\Delta t}{\tau})U(t) + \frac{\Delta t}{\tau} I_{\rm in}(t)R \tag{1}

Now assume there is no input current, :math:`I_{\rm in}(t)=0 A`:

.. math:: U(t+\Delta t) = (1-\frac{\Delta t}{\tau})U(t) \tag{2}

Let the ratio of subsequent values of :math:`U`, i.e.,
:math:`U(t+\Delta t)/U(t)` be the decay rate of the membrane potential,
also known as the inverse time constant:

.. math:: U(t+\Delta t) = \beta U(t) \tag{3}

From :math:`(1)`, this implies that:

.. math:: \beta = (1-\frac{\Delta t}{\tau}) \tag{4}

For reasonable accuracy, :math:`\Delta t << \tau`.

1.2 Weighted Input Current
^^^^^^^^^^^^^^^^^^^^^^^^^^

If we assume :math:`t` represents time-steps in a sequence rather than
continuous time, then we can set :math:`\Delta t = 1`. To
further reduce the number of hyperparameters, assume :math:`R=1`. From
:math:`(4)`, these assumptions lead to:

.. math:: \beta = (1-\frac{1}{C}) \implies (1-\beta)I_{\rm in} = \frac{1}{\tau}I_{\rm in} \tag{5}

The input current is weighted by :math:`(1-\beta)`. 
By additionally assuming input current instantaneously contributes to the membrane potential:

.. math:: U[t+1] = \beta U[t] + (1-\beta)I_{\rm in}[t+1] \tag{6}

Note that the discretization of time means we are assuming that each
time bin :math:`t` is brief enough such that a neuron may only emit a
maximum of one spike in this interval.

In deep learning, the weighting factor of an input is often a learnable
parameter. Taking a step away from the physically viable assumptions
made thus far, we subsume the effect of :math:`(1-\beta)` from
:math:`(6)` into a learnable weight :math:`W`, and replace
:math:`I_{\rm in}[t]` accordingly with an input :math:`X[t]`:

.. math:: WX[t] = I_{\rm in}[t] \tag{7}

This can be interpreted in the following way. :math:`X[t]` is an input
voltage, or spike, and is scaled by the synaptic conductance of
:math:`W` to generate a current injection to the neuron. This gives us
the following result:

.. math:: U[t+1] = \beta U[t] + WX[t+1] \tag{8}

In future simulations, the effects of :math:`W` and :math:`\beta` are decoupled.
:math:`W` is a learnable parameter that is updated independently of :math:`\beta`.

1.3 Spiking and Reset
^^^^^^^^^^^^^^^^^^^^^

We now introduce the spiking and reset mechanisms. Recall that if
the membrane exceeds the threshold, then the neuron emits an output
spike:

.. math::

   S[t] = \begin{cases} 1, &\text{if}~U[t] > U_{\rm thr} \\
   0, &\text{otherwise} \end{cases}

.. math::
   
   \tag{9}

If a spike is triggered, the membrane potential should be reset. The
*reset-by-subtraction* mechanism is modeled by:

.. math:: U[t+1] = \underbrace{\beta U[t]}_\text{decay} + \underbrace{WX[t+1]}_\text{input} - \underbrace{S[t]U_{\rm thr}}_\text{reset} \tag{10}

As :math:`W` is a learnable parameter, and :math:`U_{\rm thr}` is often
just set to :math:`1` (though can be tuned), this leaves the decay rate
:math:`\beta` as the only hyperparameter left to be specified. This
completes the painful part of this tutorial.

.. note::

   Some implementations might make slightly different assumptions.
   E.g., :math:`S[t] \rightarrow S[t+1]` in :math:`(9)`, or
   :math:`X[t] \rightarrow X[t+1]` in :math:`(10)`. This above
   derivation is what is used in snnTorch as we find it maps intuitively
   to a recurrent neural network representation, without any change in
   performance.

1.4 Code Implementation
^^^^^^^^^^^^^^^^^^^^^^^

Implementing this neuron in Python looks like this:

::

    def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
      spk = (mem > threshold) # if membrane exceeds threshold, spk=1, else, 0
      mem = beta * mem + w*x - spk*threshold
      return spk, mem

To set :math:`\beta`, we have the option of either using Equation
:math:`(3)` to define it, or hard-coding it directly. Here, we will use
:math:`(3)` for the sake of a demonstration, but in future, it will just be hard-coded as we
are more focused on something that works rather than biological precision.

Equation :math:`(3)` tells us that :math:`\beta` is the ratio of
membrane potential across two subsequent time steps. Solve
this using the continuous time-dependent form of the equation (assuming
no current injection), which was derived in `Tutorial
2 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__:

.. math:: U(t) = U_0e^{-\frac{t}{\tau}}

:math:`U_0` is the initial membrane potential at :math:`t=0`. Assume the
time-dependent equation is computed at discrete steps of
:math:`t, (t+\Delta t), (t+2\Delta t)~...~`, then we can find the ratio
of membrane potential between subsequent steps using:

.. math:: \beta = \frac{U_0e^{-\frac{t+\Delta t}{\tau}}}{U_0e^{-\frac{t}{\tau}}} = \frac{U_0e^{-\frac{t + 2\Delta t}{\tau}}}{U_0e^{-\frac{t+\Delta t}{\tau}}} =~~...

.. math:: \implies \beta = e^{-\frac{\Delta t}{\tau}} 

::

    # set neuronal parameters
    delta_t = torch.tensor(1e-3)
    tau = torch.tensor(5e-3)
    beta = torch.exp(-delta_t/tau)
   
::

    >>> print(f"The decay rate is: {beta:.3f}")
    The decay rate is: 0.819

Run a quick simulation to check the neuron behaves correctly in
response to a step voltage input:

::

    num_steps = 200
    
    # initialize inputs/outputs + small step current input
    x = torch.cat((torch.zeros(10), torch.ones(190)*0.5), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron parameters
    w = 0.4
    beta = 0.819
    
    # neuron simulation
    for step in range(num_steps):
      spk, mem = leaky_integrate_and_fire(mem, x[step], w=w, beta=beta)
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(x*w, mem_rec, spk_rec, thr_line=1,ylim_max1=0.5,
                     title="LIF Neuron Model With Weighted Step Voltage")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/lif_step.png?raw=true
        :align: center
        :width: 400


2. Leaky Neuron Model in snnTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This same thing can be achieved by instantiating ``snn.Leaky``, in a
similar way to how we used ``snn.Lapicque`` in the previous tutorial, but with less hyperparameters:

::

    lif1 = snn.Leaky(beta=0.8)

The neuron model is now stored in ``lif1``. To use this neuron:

**Inputs** 

* ``cur_in``: each element of :math:`W\times X[t]` is sequentially passed as an input 
* ``mem``: the previous step membrane potential, :math:`U[t-1]`, is also passed as input.

**Outputs** 

* ``spk_out``: output spike :math:`S[t]` (‘1’ if there is a spike; ‘0’ if there is no spike) 
* ``mem``: membrane potential :math:`U[t]` of the present step

These all need to be of type ``torch.Tensor``. Note that here, we assume
the input current has already been weighted before passing into the
``snn.Leaky`` neuron. This will make more sense when we construct a
network-scale model. Also, equation :math:`(10)` has been time-shifted
back one step without loss of generality.

::

    # Small step current input
    w=0.21
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*w), 0)
    mem = torch.zeros(1)
    spk = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron simulation
    for step in range(num_steps):
      spk, mem = lif1(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max1=0.5,
                     title="snn.Leaky Neuron Model")

Compare this plot with the manually derived leaky integrate-and-fire neuron. 
The membrane potential reset is slightly weaker: i.e., it uses a *soft reset*. 
This has been done intentionally because it enables better performance on a few deep learning benchmarks. 
The equation used instead is:

.. math:: U[t+1] = \underbrace{\beta U[t]}_\text{decay} + \underbrace{WX[t+1]}_\text{input} - \underbrace{\beta S[t]U_{\rm thr}}_\text{soft reset} \tag{11}


This model has the same optional input arguments of ``reset_mechanism``
and ``threshold`` as described for Lapicque’s neuron model.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/snn.leaky_step.png?raw=true
        :align: center
        :width: 450


3. A Feedforward Spiking Neural Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far, we have only considered how a single neuron responds to input
stimulus. snnTorch makes it straightforward to scale this up to a deep
neural network. In this section, we will create a 3-layer fully-connected neural
network of dimensions 784-1000-10. Compared to our simulations so far, each neuron will now integrate over
many more incoming input spikes.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_8_fcn.png?raw=true
        :align: center
        :width: 600



PyTorch is used to form the connections between neurons, and
snnTorch to create the neurons. First, initialize all layers.

::

    # layer parameters
    num_inputs = 784
    num_hidden = 1000
    num_outputs = 10
    beta = 0.99
    
    # initialize layers
    fc1 = nn.Linear(num_inputs, num_hidden)
    lif1 = snn.Leaky(beta=beta)
    fc2 = nn.Linear(num_hidden, num_outputs)
    lif2 = snn.Leaky(beta=beta)

Next, initialize the hidden variables and outputs of each spiking
neuron. As networks increase in size, this becomes more tedious.
The static method ``init_leaky()`` can be used to take care of
this. All neurons in snnTorch have their own initialization methods that
follow this same syntax, e.g., ``init_lapicque()``. The shape of the
hidden states are automatically initialized based on the input data
dimensions during the first forward pass.

::

    # Initialize hidden states
    mem1 = lif1.init_leaky()
    mem2 = lif2.init_leaky()
    
    # record outputs
    mem2_rec = []
    spk1_rec = []
    spk2_rec = []

Create an input spike train to pass to the network. There are 200 time
steps to simulate across 784 input neurons, i.e., the input originally
has dimensions of :math:`200 \times 784`. However, neural nets typically process data in minibatches. 
snnTorch, uses time-first dimensionality:

[:math:`time \times batch\_size \times feature\_dimensions`]

So ‘unsqueeze’ the input along ``dim=1`` to indicate ‘one batch’
of data. The dimensions of this input tensor must be 200 :math:`\times`
1 :math:`\times` 784:

::

    spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
    >>> print(f"Dimensions of spk_in: {spk_in.size()}")
    "Dimensions of spk_in: torch.Size([200, 1, 784])"

Now it’s finally time to run a full simulation. An intuitive way to
think about how PyTorch and snnTorch work together is that PyTorch
routes the neurons together, and snnTorch loads the results into spiking
neuron models. In terms of coding up a network, these spiking neurons
can be treated like time-varying activation functions.

Here is a sequential account of what’s going on:

-  The :math:`i^{th}` input from ``spk_in`` to the :math:`j^{th}` neuron 
   is weighted by the parameters initialized in ``nn.Linear``:
   :math:`X_{i} \times W_{ij}`
-  This generates the input current term from Equation :math:`(10)`,
   contributing to :math:`U[t+1]` of the spiking neuron
-  If :math:`U[t+1] > U_{\rm thr}`, then a spike is triggered from this
   neuron
-  This spike is weighted by the second layer weight, and the above
   process is repeated for all inputs, weights, and neurons.
-  If there is no spike, then nothing is passed to the post-synaptic
   neuron.

The only difference from our simulations thus far is that we are now
scaling the input current with a weight generated by ``nn.Linear``,
rather than manually setting :math:`W` ourselves.

::

    # network simulation
    for step in range(num_steps):
        cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
        spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
        cur2 = fc2(spk1)
        spk2, mem2 = lif2(cur2, mem2)
    
        mem2_rec.append(mem2)
        spk1_rec.append(spk1)
        spk2_rec.append(spk2)
    
    # convert lists to tensors
    mem2_rec = torch.stack(mem2_rec)
    spk1_rec = torch.stack(spk1_rec)
    spk2_rec = torch.stack(spk2_rec)
    
    plot_snn_spikes(spk_in, spk1_rec, spk2_rec, "Fully Connected Spiking Neural Network")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/mlp_raster.png?raw=true
        :align: center
        :width: 450

At this stage, the spikes don’t have any real meaning. The inputs and
weights are all randomly initialized, and no training has taken place.
But the spikes should appear to be propagating from the first layer
through to the output. If you are not seeing any spikes, then you might have
 been unlucky in the weight initialization lottery - you might want
to try re-running the last four code-blocks.

``spikeplot.spike_count`` can create a spike counter of
the output layer. The following animation will take some time to
generate.

   Note: if you are running the notebook locally on your desktop, please
   uncomment the line below and modify the path to your ffmpeg.exe

::

    from IPython.display import HTML
    
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
    spk2_rec = spk2_rec.squeeze(1).detach().cpu()
    
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'
    
    #  Plot spike count histogram
    anim = splt.spike_count(spk2_rec, fig, ax, labels=labels, animate=True)
    HTML(anim.to_html5_video())
    # anim.save("spike_bar.mp4")

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/spike_bar.mp4?raw=true"></video>
  </center>

``spikeplot.traces`` lets you visualize the membrane potential traces. We will plot 9 out of 10 output neurons. 
Compare it to the animation and raster plot above to see if you can match the traces to the neuron.

::

    # plot membrane potential traces
    splt.traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
    fig = plt.gcf() 
    fig.set_size_inches(8, 6)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/traces.png?raw=true
        :align: center
        :width: 450

It is fairly normal if some neurons are firing while others are
completely dead. Again, none of these spikes have any real meaning until
the weights have been trained.

Conclusion
^^^^^^^^^^

That covers how to simplify the leaky integrate-and-fire neuron model,
and then using it to build a spiking neural network. In practice, we
will almost always prefer to use ``snn.Leaky`` over ``snn.Lapicque`` for
training networks, as there is a smaller hyperparameter search space.

`Tutorial
4 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__
goes into detail with the 2nd-order ``snn.Synaptic`` and ``snn.Alpha``
models. This next tutorial is not necessary for training a network, so if you wish to go straight
to deep learning with snnTorch, then skip ahead to `Tutorial
5 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.

For reference, the documentation `can be found
here <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__.

Further Reading
^^^^^^^^^^^^^^^

-  `Check out the snnTorch GitHub project here. <https://github.com/jeshraghian/snntorch>`__
-  `snnTorch
   documentation <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__
   of the Lapicque, Leaky, Synaptic, and Alpha models
-  `Neuronal Dynamics: From single neurons to networks and models of
   cognition <https://neuronaldynamics.epfl.ch/index.html>`__ by Wulfram
   Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.
-  `Theoretical Neuroscience: Computational and Mathematical Modeling of
   Neural
   Systems <https://mitpress.mit.edu/books/theoretical-neuroscience>`__
   by Laurence F. Abbott and Peter Dayan


===========================
Tutorial 4 - 2nd Order Spiking Neuron Models
===========================

Tutorial written by Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_4_advanced_neurons.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_4_advanced_neurons.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_



Introduction
^^^^^^^^^^^^

In this tutorial, you will: 

* Learn about the more advanced leaky integrate-and-fire (LIF) neuron models available: ``Synaptic`` and ``Alpha``

Install the latest PyPi distribution of snnTorch.

::

    $ pip install snntorch

::

    # imports
    import snntorch as snn
    from snntorch import spikeplot as splt
    from snntorch import spikegen
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt


1. Synaptic Conductance-based LIF Neuron Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The neuron models explored in previous tutorials assume that an input voltage
spike leads to an instantaneous jump in synaptic current, which then
contributes to the membrane potential. In reality, a spike will result
in the *gradual* release of neurotransmitters from the pre-synaptic
neuron to the post-synaptic neuron. The synaptic conductance-based LIF
model accounts for the gradual temporal dynamics of input current.

1.1 Modeling Synaptic Current
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a pre-synaptic neuron fires, the voltage spike is transmitted down
the axon of the neuron. It triggers the vesicles to release
neurotransmitters into the synaptic cleft. These activate the
post-synaptic receptors, which directly influence the effective current
that flows into the post-synaptic neuron. Shown below are two types of
excitatory receptors, AMPA and NMDA.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_6_synaptic.png?raw=true
        :align: center
        :width: 600

The simplest model of synaptic current assumes an increasing current on
a very fast time-scale, followed by a relatively slow exponential decay,
as seen in the AMPA receptor response above. This is very similar to the
membrane potential dynamics of Lapicque’s model.

The synaptic model has two exponentially decaying terms:
:math:`I_{\rm syn}(t)` and :math:`U_{\rm mem}(t)`. The ratio between
subsequent terms (i.e., decay rate) of :math:`I_{\rm syn}(t)` is set to
:math:`\alpha`, and that of :math:`U(t)` is set to :math:`\beta`:

.. math::  \alpha = e^{-\Delta t/\tau_{\rm syn}}

.. math::  \beta = e^{-\Delta t/\tau_{\rm mem}}

where the duration of a single time step is normalized to
:math:`\Delta t = 1` in future. :math:`\tau_{\rm syn}` models the time
constant of the synaptic current in an analogous way to how
:math:`\tau_{\rm mem}` models the time constant of the membrane
potential. :math:`\beta` is derived in the exact same way as the
previous tutorial, with a similar approach to
:math:`\alpha`:

.. math:: I_{\rm syn}[t+1]=\underbrace{\alpha I_{\rm syn}[t]}_\text{decay} + \underbrace{WX[t+1]}_\text{input}

.. math:: U[t+1] = \underbrace{\beta U[t]}_\text{decay} + \underbrace{I_{\rm syn}[t+1]}_\text{input} - \underbrace{R[t]}_\text{reset}

The same conditions for spiking as the previous LIF neurons still hold:

.. math::

   S_{\rm out}[t] = \begin{cases} 1, &\text{if}~U[t] > U_{\rm thr} \\
   0, &\text{otherwise}\end{cases}

1.2 Synaptic Neuron Model in snnTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The synaptic condutance-based neuron model combines the synaptic current
dynamics with the passive membrane. It must be instantiated with two
input arguments: 

* :math:`\alpha`: the decay rate of the synaptic current 
* :math:`\beta`: the decay rate of the membrane potential (as with Lapicque)

::

    # Temporal dynamics
    alpha = 0.9
    beta = 0.8
    num_steps = 200
    
    # Initialize 2nd-order LIF neuron
    lif1 = snn.Synaptic(alpha=alpha, beta=beta)

Using this neuron is the exact same as previous LIF neurons, but now
with the addition of synaptic current ``syn`` as an input and output:

**Inputs** 

* ``spk_in``: each weighted input voltage spike :math:`WX[t]` is sequentially passed in 
* ``syn``: synaptic current :math:`I_{\rm syn}[t-1]` at the previous time step 
* ``mem``: membrane potential :math:`U[t-1]` at the previous time step

**Outputs** 

* ``spk_out``: output spike :math:`S[t]` (‘1’ if there is a spike; ‘0’ if there is no spike) 
* ``syn``: synaptic current :math:`I_{\rm syn}[t]` at the present time step 
* ``mem``: membrane potential :math:`U[t]` at the present time step

These all need to be of type ``torch.Tensor``. Note that the neuron
model has been time-shifted back one step without loss of generality.

Apply a periodic spiking input to see how current and membrane evolve
with time:

::

    # Periodic spiking input, spk_in = 0.2 V
    w = 0.2
    spk_period = torch.cat((torch.ones(1)*w, torch.zeros(9)), 0)
    spk_in = spk_period.repeat(20)
    
    # Initialize hidden states and output
    syn, mem = lif1.init_synaptic()
    spk_out = torch.zeros(1) 
    syn_rec = []
    mem_rec = []
    spk_rec = []
    
    # Simulate neurons
    for step in range(num_steps):
      spk_out, syn, mem = lif1(spk_in[step], syn, mem)
      spk_rec.append(spk_out)
      syn_rec.append(syn)
      mem_rec.append(mem)
    
    # convert lists to tensors
    spk_rec = torch.stack(spk_rec)
    syn_rec = torch.stack(syn_rec)
    mem_rec = torch.stack(mem_rec)
    
    plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, 
                         "Synaptic Conductance-based Neuron Model With Input Spikes")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial4/_static/syn_cond_spk.png?raw=true
        :align: center
        :width: 450

This model also has the optional input arguments of ``reset_mechanism``
and ``threshold`` as described for Lapicque’s neuron model. In summary,
each spike contributes a shifted exponential decay to the synaptic
current :math:`I_{\rm syn}`, which are all summed together. This current
is then integrated by the passive membrane equation derived in
`Tutorial 2 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_, thus generating output spikes. An illustration of this
process is provided below.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_7_stein.png?raw=true
        :align: center
        :width: 450

1.3 1st-Order vs. 2nd-Order Neurons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A natural question that arises is - *when do I want to use a 1st order
LIF neuron and when should I use this 2nd order LIF neuron?* While this
has not really been settled, my own experiments have given me some
intuition that might be useful.

**When 2nd-order neurons are better** 

* If the temporal relations of your input data occur across long time-scales, 
* or if the input spiking pattern is sparse

By having two recurrent equations with two decay terms (:math:`\alpha`
and :math:`\beta`), this neuron model is able to ‘sustain’ input spikes
over a longer duration. This can be beneficial to retaining long-term
relationships.

An alternative use case might also be:

-  When temporal codes matter

If you care for the precise timing of a spike, it seems easier to
control that for a 2nd-order neuron. In the ``Leaky`` model, a spike
would be triggered in direct synchrony with the input. For 2nd-order
models, the membrane potential is ‘smoothed out’ (i.e., the synaptic
current model low-pass filters the membrane potential), which means one
can use a finite rise time for :math:`U[t]`. This is clear in the
previous simulation, where the output spikes experience a delay with
respect to the input spikes.

**When 1st-order neurons are better** 

* Any case that doesn’t fall into the above, and sometimes, the above cases.

By having one less equation in 1st-order neuron models (such as
``Leaky``), the backpropagation process is made a little simpler. Though
having said that, the ``Synaptic`` model is functionally equivalent to
the ``Leaky`` model for :math:`\alpha=0.` In my own hyperparameter
sweeps on simple datasets, the optimal results seem to push
:math:`\alpha` as close to 0 as possible. As data increases in
complexity, :math:`\alpha` may grow larger.

2. Alpha Neuron Model (Hacked Spike Response Model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A recursive version of the Spike Response Model (SRM), or the ‘Alpha’
neuron, is also available, called using ``snn.Alpha``. The neuron models
thus far have all been based on the passive membrane model, using
ordinary differential equations to describe their dynamics.

The SRM family of models, on the other hand, is interpreted in terms of
a filter. Upon the arrival of an input spike, this spike is convolved
with the filter to give the membrane potential response. The form of
this filter can be exponential, as is the case with Lapicque’s neuron,
or they can be more complex such as a sum of exponentials. SRM models
are appealing as they can arbitrarily add refractoriness, threshold
adaptation, and any number of other features simply by embedding them
into the filter.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/exp.gif?raw=true
        :align: right
        :width: 400

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/alpha.gif?raw=true
        :align: right
        :width: 400


2.1 Modelling the Alpha Neuron Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Formally, this process is represented by:

.. math:: U_{\rm mem}(t) = \sum_i W(\epsilon * S_{\rm in})(t)

where the incoming spikes :math:`S_{\rm in}` are convolved with a spike
response kernel :math:`\epsilon( \cdot )`. The spike response is scaled
by a synaptic weight, :math:`W`. In top figure, the kernel
is an exponentially decaying function and would be the equivalent of
Lapicque’s 1st-order neuron model. On the bottom, the kernel is an alpha
function:

.. math:: \epsilon(t) = \frac{t}{\tau}e^{1-t/\tau}\Theta(t)

where :math:`\tau` is the time constant of the alpha kernel and
:math:`\Theta` is the Heaviside step function. Most kernel-based methods
adopt the alpha function as it provides a time-delay that is useful for
temporal codes that are concerned with specifying the exact spike time
of a neuron.

In snnTorch, the spike response model is not directly implemented as a
filter. Instead, it is recast into a recursive form such that only the
previous time step of values are required to calculate the next set of
values. This reduces the memory required.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_9_alpha.png?raw=true
        :align: center
        :width: 550

As the membrane potential is now determined by the sum of two
exponentials, each of these exponents has their own independent decay
rate. :math:`\alpha` defines the decay rate of the positive exponential,
and :math:`\beta` defines the decay rate of the negative exponential.

::

    alpha = 0.8
    beta = 0.7
    
    # initialize neuron
    lif2 = snn.Alpha(alpha=alpha, beta=beta, threshold=0.5)

Using this neuron is the same as the previous neurons, but the sum of
two exponential functions requires the synaptic current ``syn`` to be
split into a ``syn_exc`` and ``syn_inh`` component:

**Inputs** 

* ``spk_in``: each weighted input voltage spike :math:`WX[t]` is sequentially passed in 
* ``syn_exc``: excitatory post-synaptic current :math:`I_{\rm syn-exc}[t-1]` at the previous time step 
* ``syn_inh``: inhibitory post-synaptic current :math:`I_{\rm syn-inh}[t-1]` at the previous time step 
* ``mem``: membrane potential :math:`U_{\rm mem}[t-1]` at the present time :math:`t` at the previous time step

**Outputs** 

* ``spk_out``: output spike :math:`S_{\rm out}[t]` at the present time step (‘1’ if there is a spike; ‘0’ if there is no spike) 
* ``syn_exc``: excitatory post-synaptic :math:`I_{\rm syn-exc}[t]` at the present time step :math:`t` 
* ``syn_inh``: inhibitory post-synaptic current :math:`I_{\rm syn-inh}[t]` at the present time step :math:`t` 
* ``mem``: membrane potential :math:`U_{\rm mem}[t]` at the present time step

As with all other neuron models, these must be of type ``torch.Tensor``.

::

    # input spike: initial spike, and then period spiking 
    w = 0.85
    spk_in = (torch.cat((torch.zeros(10), torch.ones(1), torch.zeros(89), 
                         (torch.cat((torch.ones(1), torch.zeros(9)),0).repeat(10))), 0) * w).unsqueeze(1)
    
    # initialize parameters
    syn_exc, syn_inh, mem = lif2.init_alpha()
    mem_rec = []
    spk_rec = []
    
    # run simulation
    for step in range(num_steps):
      spk_out, syn_exc, syn_inh, mem = lif2(spk_in[step], syn_exc, syn_inh, mem)
      mem_rec.append(mem.squeeze(0))
      spk_rec.append(spk_out.squeeze(0))
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_spk_mem_spk(spk_in, mem_rec, spk_rec, "Alpha Neuron Model With Input Spikes")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial4/_static/alpha.png?raw=true
        :align: center
        :width: 500

As with the Lapicque and Synaptic models, the Alpha model also has
options to modify the threshold and reset mechanism.

2.2 Practical Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned for the Synaptic neuron, the more complex a model, the more
complex the backpropagation process during training. In my own
experiments, I have yet to find a case where the Alpha neuron
outperforms the Synaptic and Leaky neuron models. It seems as though
learning through a positive and negative exponential only makes the
gradient calculation process more difficult, and offsets any potential
benefits in more complex neuronal dynamics.

However, when an SRM model is expressed as a time-varying kernel (rather
than a recursive model as is done here), it seems to perform just as
well as the simpler neuron models. As an example, see the following
paper:

   `Sumit Bam Shrestha and Garrick Orchard, “SLAYER: Spike layer error
   reassignment in time”, Proceedings of the 32nd International
   Conference on Neural Information Processing Systems, pp. 1419-1328,
   2018. <https://arxiv.org/abs/1810.08646>`__

The Alpha neuron has been included with the intent of providing an
option for porting across SRM-based models over into snnTorch, although
natively training them seems to not be too effective in snnTorch. 

Conclusion
^^^^^^^^^^

We have covered all LIF neuron models available in snnTorch. As a quick
summary:

-  **Lapicque**: a physically accurate model based directly on
   RC-circuit parameters
-  **Leaky**: a simplified 1st-order model
-  **Synaptic**: a 2nd-order model that accounts for synaptic current
   evolution
-  **Alpha**: a 2nd-order model where the membrane potential tracks an
   alpha function

In general, ``Leaky`` and ``Synaptic`` seem to be the most useful for
training a network. ``Lapicque`` is good for demonstrating physically
precise models, while ``Alpha`` is only intended to capture the
behaviour of SRM neurons.

Building a network using these slighty more advanced neurons follows the
exact same procedure as in `Tutorial 3 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.

For reference, the documentation `can be found
here <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__.

Further Reading
^^^^^^^^^^^^^^^

-  `Check out the snnTorch GitHub project here. <https://github.com/jeshraghian/snntorch>`__
-  `snnTorch
   documentation <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__
   of the Lapicque, Leaky, Synaptic, and Alpha models
-  `Neuronal Dynamics: From single neurons to networks and models of
   cognition <https://neuronaldynamics.epfl.ch/index.html>`__ by Wulfram
   Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.
-  `Theoretical Neuroscience: Computational and Mathematical Modeling of
   Neural
   Systems <https://mitpress.mit.edu/books/theoretical-neuroscience>`__
   by Laurence F. Abbott and Peter Dayan


===========================================================
Tutorial 5 - Training Spiking Neural Networks with snntorch
===========================================================

Tutorial written by Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. arXiv preprint arXiv:2109.12894,
    September 2021. <https://arxiv.org/abs/2109.12894>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


Introduction
^^^^^^^^^^^^

In this tutorial, you will: 

* Learn how spiking neurons are implemented as a recurrent network 
* Understand backpropagation through time, and the associated challenges in SNNs such as the non-differentiability of spikes 
* Train a fully-connected network on the static MNIST dataset


..

   Part of this tutorial was inspired by Friedemann Zenke’s extensive
   work on SNNs. Check out his repo on surrogate gradients
   `here <https://github.com/fzenke/spytorch>`__, and a favourite paper
   of mine: E. O. Neftci, H. Mostafa, F. Zenke, `Surrogate Gradient
   Learning in Spiking Neural Networks: Bringing the Power of
   Gradient-based optimization to spiking neural
   networks. <https://ieeexplore.ieee.org/document/8891809>`__ IEEE
   Signal Processing Magazine 36, 51–63.

At the end of the tutorial, a basic supervised learning algorithm will
be implemented. We will use the original static MNIST dataset and train
a multi-layer fully-connected spiking neural network using gradient
descent to perform image classification.

Install the latest PyPi distribution of snnTorch:

::

    $ pip install snntorch

::

    # imports
    import snntorch as snn
    from snntorch import spikeplot as splt
    from snntorch import spikegen
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

1. A Recurrent Representation of SNNs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In `Tutorial 3 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_, we derived a recursive representation of a leaky
integrate-and-fire (LIF) neuron:

.. math:: U[t+1] = \underbrace{\beta U[t]}_\text{decay} + \underbrace{WX[t+1]}_\text{input} - \underbrace{R[t]}_\text{reset} \tag{1}

where input synaptic current is interpreted as
:math:`I_{\rm in}[t] = WX[t]`, and :math:`X[t]` may be some arbitrary
input of spikes, a step/time-varying voltage, or unweighted
step/time-varying current. Spiking is represented with the following
equation, where if the membrane potential exceeds the threshold, a spike
is emitted:

.. math::

   S[t] = \begin{cases} 1, &\text{if}~U[t] > U_{\rm thr} \\
   0, &\text{otherwise}\end{cases} 

.. math::
   \tag{2}

This formulation of a spiking neuron in a discrete, recursive form is
almost perfectly poised to take advantage of the developments in
training recurrent neural networks (RNNs) and sequence-based models.
This is illustrated using an *implicit* recurrent connection for the
decay of the membrane potential, and is distinguished from *explicit*
recurrence where the output spike :math:`S_{\rm out}` is fed back to the
input. In the figure below, the connection weighted by :math:`-U_{\rm thr}` 
represents the reset mechanism :math:`R[t]`.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial5/unrolled_2.png?raw=true
        :align: center
        :width: 600

The benefit of an unrolled graph is that it provides an explicit
description of how computations are performed. The process of unfolding
illustrates the flow of information forward in time (from left to right)
to compute outputs and losses, and backward in time to compute
gradients. The more time steps that are simulated, the deeper the graph
becomes.

Conventional RNNs treat :math:`\beta` as a learnable parameter.
This is also possible for SNNs, though by default, they are treated as
hyperparameters. This replaces the vanishing and exploding gradient
problems with a hyperparameter search. A future tutorial will describe how to
make :math:`\beta` a learnable parameter.

2. The Non-Differentiability of Spikes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.1 Training Using the Backprop Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An alternative way to represent the relationship between :math:`S` and
:math:`U` in :math:`(2)` is:

.. math:: S[t] = \Theta(U[t] - U_{\rm thr}) \tag{3}

where :math:`\Theta(\cdot)` is the Heaviside step function:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/3_2_spike_descrip.png?raw=true
        :align: center
        :width: 600

Training a network in this form poses some serious challenges. Consider a single, isolated time step of the computational
graph from the previous figure titled *"Recurrent representation of spiking neurons"*, as
shown in the *forward pass* below:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial5/non-diff.png?raw=true
        :align: center
        :width: 400

The goal is to train the network using the gradient of the loss with
respect to the weights, such that the weights are updated to minimize
the loss. The backpropagation algorithm achieves this using the chain
rule:

.. math::

   \frac{\partial \mathcal{L}}{\partial W} = 
   \frac{\partial \mathcal{L}}{\partial S}
   \underbrace{\frac{\partial S}{\partial U}}_{\{0, \infty\}}
   \frac{\partial U}{\partial I}\
   \frac{\partial I}{\partial W}\ \tag{4}

From :math:`(1)`,  :math:`\partial I/\partial W=X`, and
:math:`\partial U/\partial I=1`. While a loss function is yet to be defined, 
we can assume :math:`\partial \mathcal{L}/\partial S` has an
analytical solution, in a similar form to the cross-entropy or
mean-square error loss (more on that shortly).

However, the term that we are going to grapple with is
:math:`\partial S/\partial U`. The derivative of the
Heaviside step function from :math:`(3)` is the Dirac Delta
function, which evaluates to :math:`0` everywhere, except at the threshold
:math:`U_{\rm thr} = \theta`, where it tends to infinity. This means the
gradient will almost always be nulled to zero (or saturated if :math:`U`
sits precisely at the threshold), and no learning can take place. This
is known as the **dead neuron problem**.

2.2 Overcoming the Dead Neuron Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common way to address the dead neuron problem is to keep the
Heaviside function as it is during the forward pass, but swap the
derivative term :math:`\partial S/\partial U` for something that does
not kill the learning process during the backward pass, which will be
denoted :math:`\partial \tilde{S}/\partial U`. This might sound odd, but
it turns out that neural networks are quite robust to such
approximations. This is commonly known as the *surrogate gradient*
approach.

A variety of options exist to using surrogate gradients, and we will
dive into more detail on these methods in `Tutorial 6 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_. 
The default method in snnTorch (as of v0.6.0) is to smooth the Heaviside function with the arctangent function. 
The backward-pass derivative used is:


.. math::

    \frac{\partial \tilde{S}}{\partial U} \leftarrow \frac{1}{\pi}\frac{1}{(1+[U\pi]^2)}


where the left arrow denotes substitution.

The same neuron model described in :math:`(1)-(2)` (a.k.a.,
``snn.Leaky`` neuron from Tutorial 3) is implemented in PyTorch below.
Don’t worry if you don’t understand this. This will be
condensed into one line of code using snnTorch in a moment:

::

    # Leaky neuron model, overriding the backward pass with a custom function
    class LeakySurrogate(nn.Module):
      def __init__(self, beta, threshold=1.0):
          super(LeakySurrogate, self).__init__()
    
          # initialize decay rate beta and threshold
          self.beta = beta
          self.threshold = threshold
          self.spike_gradient = self.ATan.apply
      
      # the forward function is called each time we call Leaky
      def forward(self, input_, mem):
        spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
        reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
        mem = self.beta * mem + input_ - reset  # Eq (1)
        return spk, mem
    
      # Forward pass: Heaviside function
      # Backward pass: Override Dirac Delta with the derivative of the ArcTan function 
      @staticmethod
      class ATan(torch.autograd.Function):
          @staticmethod
          def forward(ctx, mem):
              spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
              ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
              return spk
    
          @staticmethod
          def backward(ctx, grad_output):
              (spk,) = ctx.saved_tensors  # retrieve the membrane potential 
              grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
              return grad

Note that the reset mechanism is detached from the computational graph, as the surrogate gradient should only be applied to :math:`\partial S/\partial U`, and not :math:`\partial R/\partial U`.

The above neuron is instantiated using:

::

    lif1 = LeakySurrogate(beta=0.9)

This neuron can be simulated using a for-loop, just as in previous
tutorials, while PyTorch’s automatic differentation (autodiff) mechanism
keeps track of the gradient in the background.

The same thing can be accomplished by calling
the ``snn.Leaky`` neuron. In fact, every time you call any neuron model
from snnTorch, the *ATan* surrogate gradient is applied to it
by default:

::

    lif1 = snn.Leaky(beta=0.9)

If you would like to explore how this neuron behaves, then refer to
`Tutorial
3 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__.

3. Backprop Through Time 
^^^^^^^^^^^^^^^^^^^^^^^^^

Equation :math:`(4)` only calculates the
gradient for one single time step (referred to as the *immediate
influence* in the figure below), but the backpropagation through time
(BPTT) algorithm calculates the gradient from the loss to *all*
descendants and sums them together.

The weight :math:`W` is applied at every time step, and so imagine a
loss is also calculated at every time step. The influence of the weight
on present and historical losses must be summed together to define the
global gradient:

.. math::

   \frac{\partial \mathcal{L}}{\partial W}=\sum_t \frac{\partial\mathcal{L}[t]}{\partial W} = 
   \sum_t \sum_{s\leq t} \frac{\partial\mathcal{L}[t]}{\partial W[s]}\frac{\partial W[s]}{\partial W} \tag{5} 

The point of :math:`(5)` is to ensure causality: by constraining
:math:`s\leq t`, we only account for the contribution of immediate and
prior influences of :math:`W` on the loss. A recurrent system constrains
the weight to be shared across all steps: :math:`W[0]=W[1] =~... ~ = W`.
Therefore, a change in :math:`W[s]` will have the same effect on all
:math:`W`, which implies that :math:`\partial W[s]/\partial W=1`:

.. math::

   \frac{\partial \mathcal{L}}{\partial W}=
   \sum_t \sum_{s\leq t} \frac{\partial\mathcal{L}[t]}{\partial W[s]} \tag{6} 

As an example, isolate the prior influence due to :math:`s = t-1` *only*; this
means the backward pass must track back in time by one step. The
influence of :math:`W[t-1]` on the loss can be written as:

.. math::

   \frac{\partial \mathcal{L}[t]}{\partial W[t-1]} = 
   \frac{\partial \mathcal{L}[t]}{\partial S[t]}
   \underbrace{\frac{\partial \tilde{S}[t]}{\partial U[t]}}_{Eq.~(5)}
   \underbrace{\frac{\partial U[t]}{\partial U[t-1]}}_\beta
   \underbrace{\frac{\partial U[t-1]}{\partial I[t-1]}}_1
   \underbrace{\frac{\partial I[t-1]}{\partial W[t-1]}}_{X[t-1]} \tag{7}

We have already dealt with all of these terms from :math:`(4)`, except
for :math:`\partial U[t]/\partial U[t-1]`. From :math:`(1)`, this
temporal derivative term simply evaluates to :math:`\beta`. So if we
really wanted to, we now know enough to painstakingly calculate the
derivative of every weight at every time step by hand, and it’d look
something like this for a single neuron:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial5/bptt.png?raw=true
        :align: center
        :width: 600


But thankfully, PyTorch’s autodiff takes care of that in the background for
us.


.. note::
  The reset mechanism has been omitted from the above figure. In snnTorch, reset is included in the forward-pass, but detached from the backward pass.

4. Setting up the Loss / Output Decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a conventional, non-spiking neural network, a supervised, multi-class
classification problem takes the neuron with the highest activation
and treats that as the predicted class.

In a spiking neural net, there are several options to interpreting the output spikes. The most common approaches are:

* **Rate coding:** Take the neuron with the highest firing rate (or spike count) as the predicted class 
* **Latency coding:** Take the neuron that fires *first* as the predicted class

This might feel familiar to `Tutorial 1 on neural
encoding <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__.
The difference is that, here, we are interpreting (decoding) the output
spikes, rather than encoding/converting raw input data into spikes.

Let’s focus on a rate code. When input data is passed to the network, we
want the correct neuron class to emit the most spikes over the course of
the simulation run. This corresponds to the highest average firing
frequency. One way to achieve this is to increase the membrane
potential of the correct class to :math:`U>U_{\rm thr}`, and that of
incorrect classes to :math:`U<U_{\rm thr}`. Applying the target to
:math:`U` serves as a proxy for modulating spiking behavior from
:math:`S`.

This can be implemented by taking the softmax of the membrane potential
for output neurons, where :math:`C` is the number of output classes:

.. math:: p_i[t] = \frac{e^{U_i[t]}}{\sum_{i=0}^{C}e^{U_i[t]}} \tag{8}

The cross-entropy between :math:`p_i` and the target
:math:`y_i \in \{0,1\}^C`, which is a one-hot target vector, is obtained
using:

.. math:: \mathcal{L}_{CE}[t] = -\sum_{i=0}^Cy_i{\rm log}(p_i[t]) \tag{9}

The practical effect is that the membrane potential of the correct class
is encouraged to increase while those of incorrect classes are reduced. In effect, this means the correct class is encouraged to fire
at all time steps, while incorrect classes are suppressed at all steps.
This may not be the most efficient implementation of an SNN, but
it is among the simplest.

This target is applied at every time step of the simulation, thus also
generating a loss at every step. These losses are then summed together
at the end of the simulation:

.. math:: \mathcal{L}_{CE} = \sum_t\mathcal{L}_{CE}[t] \tag{10}

This is just one of many possible ways to apply a loss function to a
spiking neural network. A variety of approaches are available to use in
snnTorch (in the module ``snn.functional``), and will be the subject of
a future tutorial.

With all of the background theory having been taken care of, let’s finally dive into
training a fully-connected spiking neural net.

5. Setting up the Static MNIST Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # dataloader arguments
    batch_size = 128
    data_path='/tmp/data/mnist'
    
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

::

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

::

    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

6. Define the Network
^^^^^^^^^^^^^^^^^^^^^

::

    # Network Architecture
    num_inputs = 28*28
    num_hidden = 1000
    num_outputs = 10
    
    # Temporal Dynamics
    num_steps = 25
    beta = 0.95

::

    # Define Network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
    
            # Initialize layers
            self.fc1 = nn.Linear(num_inputs, num_hidden)
            self.lif1 = snn.Leaky(beta=beta)
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            self.lif2 = snn.Leaky(beta=beta)
    
        def forward(self, x):
    
            # Initialize hidden states at t=0
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            
            # Record the final layer
            spk2_rec = []
            mem2_rec = []
    
            for step in range(num_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                mem2_rec.append(mem2)
    
            return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
            
    # Load the network onto CUDA if available
    net = Net().to(device)

The code in the ``forward()`` function will only be called once the
input argument ``x`` is explicitly passed into ``net``.

-  ``fc1`` applies a linear transformation to all input pixels from the
   MNIST dataset;
-  ``lif1`` integrates the weighted input over time, emitting a spike if
   the threshold condition is met;
-  ``fc2`` applies a linear transformation to the output spikes of
   ``lif1``;
-  ``lif2`` is another spiking neuron layer, integrating the weighted
   spikes over time.

7. Training the SNN
^^^^^^^^^^^^^^^^^^^

7.1 Accuracy Metric
^^^^^^^^^^^^^^^^^^^

Below is a function that takes a batch of data, counts up all the
spikes from each neuron (i.e., a rate code over the simulation time),
and compares the index of the highest count with the actual target. If
they match, then the network correctly predicted the target.

::

    # pass data into the network, sum the spikes over time
    # and compare the neuron with the highest number of spikes
    # with the target
    
    def print_batch_accuracy(data, targets, train=False):
        output, _ = net(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())
    
        if train:
            print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        else:
            print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
    
    def train_printer():
        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[counter]:.2f}")
        print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
        print_batch_accuracy(data, targets, train=True)
        print_batch_accuracy(test_data, test_targets, train=False)
        print("\n")

7.2 Loss Definition
^^^^^^^^^^^^^^^^^^^

The ``nn.CrossEntropyLoss`` function in PyTorch automatically handles taking
the softmax of the output layer as well as generating a loss at the
output.

::

    loss = nn.CrossEntropyLoss()

7.3 Optimizer
^^^^^^^^^^^^^

Adam is a robust optimizer that performs well on recurrent networks, so
let’s use that with a learning rate of :math:`5\times10^{-4}`.

::

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

7.4 One Iteration of Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take the first batch of data and load it onto CUDA if available.

::

    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)

Flatten the input data to a vector of size :math:`784` and pass it into
the network.

::

    spk_rec, mem_rec = net(data.view(batch_size, -1))

::

    >>> print(mem_rec.size())
    torch.Size([25, 128, 10])

The recording of the membrane potential is taken across: 

* 25 time steps 
* 128 samples of data 
* 10 output neurons

We wish to calculate the loss at every time step, and sum these up
together, as per Equation :math:`(10)`:

::

    # initialize the total loss value
    loss_val = torch.zeros((1), dtype=dtype, device=device)
    
    # sum loss at every step
    for step in range(num_steps):
      loss_val += loss(mem_rec[step], targets)

::

    >>> print(f"Training loss: {loss_val.item():.3f}")
    Training loss: 60.488

The loss is quite large, because it is summed over 25 time
steps. The accuracy is also bad (it should be roughly around 10%) as the
network is untrained:

::

    >>> print_batch_accuracy(data, targets, train=True)
    Train set accuracy for a single minibatch: 10.16%

A single weight update is applied to the network as follows:

::

      # clear previously stored gradients
      optimizer.zero_grad()
    
      # calculate the gradients
      loss_val.backward()
    
      # weight update
      optimizer.step()

Now, re-run the loss calculation and accuracy after a single
iteration:

::

    # calculate new network outputs using the same data
    spk_rec, mem_rec = net(data.view(batch_size, -1))
    
    # initialize the total loss value
    loss_val = torch.zeros((1), dtype=dtype, device=device)
    
    # sum loss at every step
    for step in range(num_steps):
      loss_val += loss(mem_rec[step], targets)

::

    >>> print(f"Training loss: {loss_val.item():.3f}")
    >>> print_batch_accuracy(data, targets, train=True)
    Training loss: 47.384
    Train set accuracy for a single minibatch: 33.59%

After only one iteration, the loss should have decreased and accuracy
should have increased. Note how membrane potential is used to calculate the cross entropy
loss, and spike count is used for the measure of accuracy. It is also possible to use the spike count in the loss (`see Tutorial 6 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_)

7.5 Training Loop
^^^^^^^^^^^^^^^^^

Let’s combine everything into a training
loop. We will train for one epoch (though feel free to increase
``num_epochs``), exposing our network to each sample of data once.

::

    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0
    
    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
    
        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)
    
            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))
    
            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)
    
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
    
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
    
            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)
    
                # Test set forward pass
                test_spk, test_mem = net(test_data.view(batch_size, -1))
    
                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())
    
                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer()
                counter += 1
                iter_counter +=1

The terminal will iteratively print out something like this every 50 iterations:

::

    Epoch 0, Iteration 50
    Train Set Loss: 12.63
    Test Set Loss: 13.44
    Train set accuracy for a single minibatch: 92.97%
    Test set accuracy for a single minibatch: 90.62%


8. Results
^^^^^^^^^^

8.1 Plot Training/Test Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # Plot Loss
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial5/loss.png?raw=true
        :align: center
        :width: 550

The loss curves are noisy because the losses are tracked at every iteration, rather than averaging across multiple iterations. 

8.2 Test Set Accuracy
^^^^^^^^^^^^^^^^^^^^^

This function iterates over all minibatches to obtain a measure of
accuracy over the full 10,000 samples in the test set.

::

    total = 0
    correct = 0
    
    # drop_last switched to False to keep all samples
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
    
    with torch.no_grad():
      net.eval()
      for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        
        # forward pass
        test_spk, _ = net(data.view(data.size(0), -1))
    
        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

::

    >>> print(f"Total correctly classified test set images: {correct}/{total}")
    >>> print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
    Total correctly classified test set images: 9387/10000
    Test Set Accuracy: 93.87%

Voila! That’s it for static MNIST. Feel free to tweak the network
parameters, hyperparameters, decay rate, using a learning rate scheduler
etc. to see if you can improve the network performance.

Conclusion
^^^^^^^^^^

Now you know how to construct and train a fully-connected network on a
static dataset. The spiking neurons can also be adapted to other
layer types, including convolutions and skip connections. Armed with
this knowledge, you should now be able to build many different types of
SNNs. `In the next
tutorial <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__,
you will learn how to train a spiking convolutional network, and simplify the amount of code required using the ``snn.backprop`` module.

Also, a special thanks to Bugra Kaytanli for providing valuable feedback on the tutorial.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.


Additional Resources 
^^^^^^^^^^^^^^^^^^^^^

- `Check out the snnTorch GitHub project here. <https://github.com/jeshraghian/snntorch>`__


===============================================================================================
Tutorial 6 - Surrogate Gradient Descent in a Convolutional SNN
===============================================================================================

Tutorial written by Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_6_CNN.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_6_CNN.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_



Introduction
^^^^^^^^^^^^

In this tutorial, you will: 

* Learn how to modify surrogate gradient descent to overcome the dead neuron problem 
* Construct and train a convolutional spiking neural network 
* Use a sequential container, ``nn.Sequential`` to simplify model construction 

..

   Part of this tutorial was inspired by Friedemann Zenke’s extensive
   work on SNNs. Check out his repo on surrogate gradients
   `here <https://github.com/fzenke/spytorch>`__, and a favourite paper
   of mine: E. O. Neftci, H. Mostafa, F. Zenke, `Surrogate Gradient
   Learning in Spiking Neural Networks: Bringing the Power of
   Gradient-based optimization to spiking neural
   networks. <https://ieeexplore.ieee.org/document/8891809>`__ IEEE
   Signal Processing Magazine 36, 51–63.


At the end of the tutorial, we will train a convolutional spiking neural
network (CSNN) using the MNIST dataset to perform image classification.
The background theory follows on from `Tutorials 2, 4 and
5 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__,
so feel free to go back if you need to brush up.

Install the latest PyPi distribution of snnTorch:

::

    $ pip install snntorch

::

    # imports
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import backprop
    from snntorch import functional as SF
    from snntorch import utils
    from snntorch import spikeplot as splt
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

1. Surrogate Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Tutorial 5 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ raised the **dead neuron problem**. This arises
because of the non-differentiability of spikes:

.. math:: S[t] = \Theta(U[t] - U_{\rm thr}) \tag{1}

.. math:: \frac{\partial S}{\partial U} = \delta(U - U_{\rm thr}) \in \{0, \infty\} \tag{2}

where :math:`\Theta(\cdot)` is the Heaviside step function, and
:math:`\delta(\cdot)` is the Dirac-Delta function. We previously
overcame this using the threshold-shifted *ArcTangent* function on the backward pass instead. 

Other common smoothing functions include the sigmoid function, or the fast
sigmoid function. The sigmoidal functions must also be shifted such that
they are centered at the threshold :math:`U_{\rm thr}.` Defining the
overdrive of the membrane potential as :math:`U_{OD} = U - U_{\rm thr}`:

.. math:: \tilde{S} = \frac{U_{OD}}{1+k|U_{OD}|} \tag{3}

.. math:: \frac{\partial \tilde{S}}{\partial U} = \frac{1}{(k|U_{OD}|+1)^2}\tag{4}

where :math:`k` modulates how smooth the surrogate function is, and is
treated as a hyperparameter. As :math:`k` increases, the approximation
converges towards the original derivative in :math:`(2)`:

.. math:: \frac{\partial \tilde{S}}{\partial U} \Bigg|_{k \rightarrow \infty} = \delta(U-U_{\rm thr})


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial6/surrogate.png?raw=true
        :align: center
        :width: 800


To summarize:

-  **Forward Pass**

   -  Determine :math:`S` using the shifted Heaviside function in
      :math:`(1)`
   -  Store :math:`U` for later use during the backward pass

-  **Backward Pass**

   -  Pass :math:`U` into :math:`(4)` to calculate the derivative term

In the same way the *ArcTangent* approach was used in `Tutorial 5 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_, 
the gradient of the fast sigmoid function can override the Dirac-Delta function in a Leaky Integrate-and-Fire
(LIF) neuron model:

::

    # Leaky neuron model, overriding the backward pass with a custom function
    class LeakySigmoidSurrogate(nn.Module):
      def __init__(self, beta, threshold=1.0, k=25):

          # Leaky_Surrogate is defined in the previous tutorial and not used here
          super(Leaky_Surrogate, self).__init__()
    
          # initialize decay rate beta and threshold
          self.beta = beta
          self.threshold = threshold
          self.surrogate_func = self.FastSigmoid.apply
      
      # the forward function is called each time we call Leaky
      def forward(self, input_, mem):
        spk = self.surrogate_func((mem-self.threshold))  # call the Heaviside function
        reset = (spk - self.threshold).detach()
        mem = self.beta * mem + input_ - reset
        return spk, mem
    
      # Forward pass: Heaviside function
      # Backward pass: Override Dirac Delta with gradient of fast sigmoid
      @staticmethod
      class FastSigmoid(torch.autograd.Function):  
        @staticmethod
        def forward(ctx, mem, k=25):
            ctx.save_for_backward(mem) # store the membrane potential for use in the backward pass
            ctx.k = k
            out = (mem > 0).float() # Heaviside on the forward pass: Eq(1)
            return out
    
        @staticmethod
        def backward(ctx, grad_output): 
            (mem,) = ctx.saved_tensors  # retrieve membrane potential
            grad_input = grad_output.clone()
            grad = grad_input / (ctx.k * torch.abs(mem) + 1.0) ** 2  # gradient of fast sigmoid on backward pass: Eq(4)
            return grad, None

Better yet, all of that can be condensed by using the built-in module
``snn.surrogate`` from snnTorch, where :math:`k` from :math:`(4)` is
denoted ``slope``. The surrogate gradient is passed into ``spike_grad``
as an argument:

::

    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    
    lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

To explore the other surrogate gradient functions available, `take a
look at the documentation
here. <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html>`__

2. Setting up the CSNN
^^^^^^^^^^^^^^^^^^^^^^

2.1 DataLoaders
^^^^^^^^^^^^^^^

::

    # dataloader arguments
    batch_size = 128
    data_path='/tmp/data/mnist'
    
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

::

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

2.2 Define the Network
^^^^^^^^^^^^^^^^^^^^^^

The convolutional network architecture to be used is:
12C5-MP2-64C5-MP2-1024FC10

-  12C5 is a 5 :math:`\times` 5 convolutional kernel with 12
   filters
-  MP2 is a 2 :math:`\times` 2 max-pooling function
-  1024FC10 is a fully-connected layer that maps 1,024 neurons to 10
   outputs

::

    # neuron and simulation parameters
    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    num_steps = 50

::

    # Define Network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
    
            # Initialize layers
            self.conv1 = nn.Conv2d(1, 12, 5)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.conv2 = nn.Conv2d(12, 64, 5)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc1 = nn.Linear(64*4*4, 10)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
        def forward(self, x):
    
            # Initialize hidden states and outputs at t=0
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky() 
            mem3 = self.lif3.init_leaky()
    
            cur1 = F.max_pool2d(self.conv1(x), 2)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.view(batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
    
            return spk3, mem3

In the previous tutorial, the network was wrapped inside of a class, as shown above. 
With increasing network complexity, this adds a
lot of boilerplate code that we might wish to avoid. Alternatively, the ``nn.Sequential`` method can be used instead.

.. note::
    The following code-block simulates over one single time-step, and requires a separate for-loop over time.

::

    #  Initialize Network
    net = nn.Sequential(nn.Conv2d(1, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 64, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(64*4*4, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)

The ``init_hidden`` argument initializes the hidden states of the neuron 
(here, membrane potential). This takes place in the background as an instance variable. 
If ``init_hidden`` is activated, the membrane potential is not explicitly returned to 
the user, ensuring only the output spikes are sequentially passed through the layers wrapped in ``nn.Sequential``. 

To train a model using the final layer's membrane potential, set the argument ``output=True``. 
This enables the final layer to return both the spike and membrane potential response of the neuron.

2.3 Forward-Pass
^^^^^^^^^^^^^^^^

A forward pass across a simulation duration of ``num_steps`` looks like
this:

::

    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)
    
    for step in range(num_steps):
        spk_out, mem_out = net(data)

Wrap that in a function, recording the membrane potential and
spike response over time:

::

    def forward_pass(net, num_steps, data):
      mem_rec = []
      spk_rec = []
      utils.reset(net)  # resets hidden states for all LIF neurons in net
    
      for step in range(num_steps):
          spk_out, mem_out = net(data)
          spk_rec.append(spk_out)
          mem_rec.append(mem_out)
      
      return torch.stack(spk_rec), torch.stack(mem_rec)

::

    spk_rec, mem_rec = forward_pass(net, num_steps, data)

3. Training Loop
^^^^^^^^^^^^^^^^

3.1 Loss Using snn.Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous tutorial, the Cross Entropy Loss between the membrane potential of the output neurons and the target was used to train the network. 
This time, the total number of spikes from each neuron will be used to calculate the Cross Entropy instead.

A variety of loss functions are included in the ``snn.functional`` module, which is analogous to ``torch.nn.functional`` in PyTorch. 
These implement a mix of cross entropy and mean square error losses, are applied to spikes and/or membrane potential, to train a rate or latency-coded network. 

The approach below applies the cross entropy loss to the output spike count in order train a rate-coded network:

::

    # already imported snntorch.functional as SF 
    loss_fn = SF.ce_rate_loss()

The recordings of the spike are passed as the first argument to
``loss_fn``, and the target neuron index as the second argument to
generate a loss. `The documentation provides further information and
exmaples. <https://snntorch.readthedocs.io/en/latest/snntorch.functional.html#snntorch.functional.ce_rate_loss>`__

::

    loss_val = loss_fn(spk_rec, targets)

::

    >>> print(f"The loss from an untrained network is {loss_val.item():.3f}")
    The loss from an untrained network is 2.303

3.2 Accuracy Using snn.Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``SF.accuracy_rate()`` function works similarly, in that the
predicted output spikes and actual targets are supplied as arguments.
``accuracy_rate`` assumes a rate code is used to interpret the output by checking if the index of the neuron with the highest spike count
matches the target index.

::

    acc = SF.accuracy_rate(spk_rec, targets)

::

    >>> print(f"The accuracy of a single batch using an untrained network is {acc*100:.3f}%")
    The accuracy of a single batch using an untrained network is 10.938%

As the above function only returns the accuracy of a single batch of
data, the following function returns the accuracy on the entire
DataLoader object:

::

    def batch_accuracy(train_loader, net, num_steps):
      with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        
        train_loader = iter(train_loader)
        for data, targets in train_loader:
          data = data.to(device)
          targets = targets.to(device)
          spk_rec, _ = forward_pass(net, num_steps, data)
    
          acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
          total += spk_rec.size(1)
    
      return acc/total

::

    test_acc = batch_accuracy(test_loader, net, num_steps)

::

    >>> print(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")
    The total accuracy on the test set is: 8.59%

3.3 Training Loop
^^^^^^^^^^^^^^^^^

The following training loop is qualitatively similar to the previous tutorial.

::

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    num_epochs = 1
    loss_hist = []
    test_acc_hist = []
    counter = 0

    # Outer training loop
    for epoch in range(num_epochs):

        # Training loop
        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, _ = forward_pass(net, num_steps, data)

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            if counter % 50 == 0:
            with torch.no_grad():
                net.eval()

                # Test set forward pass
                test_acc = batch_accuracy(test_loader, net, num_steps)
                print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                test_acc_hist.append(test_acc.item())

            counter += 1


The output should look something like this:

::

    Iteration 0, Test Acc: 9.82%

    Iteration 50, Test Acc: 91.98%

    Iteration 100, Test Acc: 94.90%

    Iteration 150, Test Acc: 95.70%


Despite having selected some fairly generic values and architectures,
the test set accuracy should be fairly competitive given the brief
training run!

4. Results
^^^^^^^^^^

4.1 Plot Test Accuracy
^^^^^^^^^^^^^^^^^^^^^^

::

    # Plot Loss
    fig = plt.figure(facecolor="w")
    plt.plot(test_acc_hist)
    plt.title("Test Set Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial6/test_acc.png?raw=true
        :align: center
        :width: 450

4.2 Spike Counter
^^^^^^^^^^^^^^^^^

Run a forward pass on a batch of data to obtain spike and membrane
readings.

::

    spk_rec, mem_rec = forward_pass(net, num_steps, data)

Changing ``idx`` allows you to index into various samples from the
simulated minibatch. Use ``splt.spike_count`` to explore the spiking
behaviour of a few different samples!

   Note: if you are running the notebook locally on your desktop, please
   uncomment the line below and modify the path to your ffmpeg.exe

::

    from IPython.display import HTML
    
    idx = 0
    
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
    
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'
    
    #  Plot spike count histogram
    anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels, 
                            animate=True, interpolate=4)
    
    HTML(anim.to_html5_video())
    # anim.save("spike_bar.mp4")


.. raw:: html

    <center>
        <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial6/spike_bar.mp4?raw=true"></video>
    </center>

::

    >>> print(f"The target label is: {targets[idx]}")
    The target label is: 3

Conclusion
^^^^^^^^^^

You should now have a grasp of the basic features of snnTorch and
be able to start running your own experiments. `In the next
tutorial <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__,
we will train a network using a neuromorphic dataset.

A special thanks to `Gianfrancesco Angelini <https://github.com/gianfa>`__ for providing valuable feedback on the tutorial.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.

Additional Resources 
^^^^^^^^^^^^^^^^^^^^^

- `Check out the snnTorch GitHub project here. <https://github.com/jeshraghian/snntorch>`__

===============================================================================================
Tutorial 7 - Neuromorphic Datasets with Tonic + snnTorch
===============================================================================================

Tutorial written by Gregor Lenz (`https://lenzgregor.com <https://lenzgregor.com)>`_) and Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


Introduction
^^^^^^^^^^^^


In this tutorial, you will: 

* Learn how to load neuromorphic datasets using `Tonic <https://github.com/neuromorphs/tonic>`__ 
* Make use of caching to speed up dataloading 
* Train a CSNN with the `Neuromorphic-MNIST <https://tonic.readthedocs.io/en/latest/datasets.html#n-mnist>`__ Dataset

Install the latest PyPi distribution of snnTorch:

::

    pip install tonic 
    pip install snntorch

1. Using Tonic to Load Neuromorphic Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Loading datasets from neuromorphic sensors is made super simple thanks
to `Tonic <https://github.com/neuromorphs/tonic>`__, which works much
like PyTorch vision.

Let’s start by loading the neuromorphic version of the MNIST dataset,
called
`N-MNIST <https://tonic.readthedocs.io/en/latest/reference/datasets.html#n-mnist>`__.
We can have a look at some raw events to get a feel for what we’re
working with.

::

    import tonic
    
    dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
    events, target = dataset[0]

::

    >>> print(events)
    [(10, 30, 937, 1) (33, 20, 1030, 1) (12, 27, 1052, 1) ...
    ( 7, 15, 302706, 1) (26, 11, 303852, 1) (11, 17, 305341, 1)]

Each row corresponds to a single event, which consists of four
parameters: (*x-coordinate, y-coordinate, timestamp, polarity*).

-  x & y co-ordinates correspond to an address in a :math:`34 \times 34`
   grid.

-  The timestamp of the event is recorded in microseconds.

-  The polarity refers to whether an on-spike (+1) or an off-spike (-1)
   occured; i.e., an increase in brightness or a decrease in brightness.

If we were to accumulate those events over time and plot the bins as
images, it looks like this:

::

    >>> tonic.utils.plot_event_grid(events)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial7/tonic_event_grid.png?raw=true
        :align: center
        :width: 450

1.1 Transformations
^^^^^^^^^^^^^^^^^^^

However, neural nets don’t take lists of events as input. The raw data
must be converted into a suitable representation, such as a tensor. We
can choose a set of transforms to apply to our data before feeding it to
our network. The neuromorphic camera sensor has a temporal resolution of
microseconds, which when converted into a dense representation, ends up
as a very large tensor. That is why we bin events into a smaller number
of frames using the `ToFrame
transformation <https://tonic.readthedocs.io/en/latest/reference/transformations.html#frames>`__,
which reduces temporal precision but also allows us to work with it in a
dense format.

-  ``time_window=1000`` integrates events into 1000\ :math:`~\mu`\ s
   bins

-  Denoise removes isolated, one-off events. If no event occurs within a
   neighbourhood of 1 pixel across ``filter_time`` microseconds, the
   event is filtered. Smaller ``filter_time`` will filter more events.

::

    import tonic.transforms as transforms
    
    sensor_size = tonic.datasets.NMNIST.sensor_size
    
    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, 
                                                             time_window=1000)
                                         ])
    
    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)



1.2 Fast DataLoading
^^^^^^^^^^^^^^^^^^^^

The original data is stored in a format that is slow to read. To speed up 
dataloading, we can make use of disk caching and batching. That means that 
once files are loaded from the original dataset, they are written to the disk. 

Because event recordings have different lengths, we are going to provide a 
collation function ``tonic.collation.PadTensors()`` that will pad out shorter 
recordings to ensure all samples in a batch have the same dimensions. 

::

    from torch.utils.data import DataLoader
    from tonic import DiskCachedDataset


    cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    cached_dataloader = DataLoader(cached_trainset)

    batch_size = 128
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

::

    def load_sample_batched():
        events, target = next(iter(cached_dataloader))

::

    >>> %timeit -o -r 10 load_sample_batched()
    4.2 ms ± 119 µs per loop (mean ± std. dev. of 10 runs, 100 loops each)


By using disk caching and a PyTorch dataloader with multithreading and batching 
support, we have signifantly reduced loading times.

If you have a large amount of RAM available, you can speed up dataloading further 
by caching to main memory instead of to disk:

::
^^
    from tonic import MemoryCachedDataset

    cached_trainset = MemoryCachedDataset(trainset)


2. Training our network using frames created from events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now let’s actually train a network on the N-MNIST classification task.
We start by defining our caching wrappers and dataloaders. While doing
that, we’re also going to apply some augmentations to the training data.
The samples we receive from the cached dataset are frames, so we can
make use of PyTorch Vision to apply whatever random transform we would
like.

::

    import torch
    import torchvision
    
    transform = tonic.transforms.Compose([torch.from_numpy,
                                          torchvision.transforms.RandomRotation([-10,10])])
    
    cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')
    
    # no augmentations for the testset
    cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')
    
    batch_size = 128
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

A mini-batch now has the dimensions (time steps, batch size, channels,
height, width). The number of time steps will be set to that of the
longest recording in the mini-batch, and all other samples will be
padded with zeros to match it.

::

    >>> event_tensor, target = next(iter(trainloader))
    >>> print(event_tensor.shape)
    torch.Size([311, 128, 2, 34, 34])


2.1 Defining our network
^^^^^^^^^^^^^^^^^^^^^^^^

We will use snnTorch + PyTorch to construct a CSNN, just as in the
previous tutorial. The convolutional network architecture to be used is:
12C5-MP2-32C5-MP2-800FC10

-  12C5 is a 5 :math:`\times` 5 convolutional kernel with 12
   filters
-  MP2 is a 2 :math:`\times` 2 max-pooling function
-  800FC10 is a fully-connected layer that maps 800 neurons to 10
   outputs


::

    import snntorch as snn
    from snntorch import surrogate
    from snntorch import functional as SF
    from snntorch import spikeplot as splt
    from snntorch import utils
    import torch.nn as nn

::

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # neuron and simulation parameters
    spike_grad = surrogate.atan()
    beta = 0.5
    
    #  Initialize Network
    net = nn.Sequential(nn.Conv2d(2, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 32, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(32*5*5, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)

::

    # this time, we won't return membrane as we don't need it 
    
    def forward_pass(net, data):  
      spk_rec = []
      utils.reset(net)  # resets hidden states for all LIF neurons in net
    
      for step in range(data.size(0)):  # data.size(0) = number of time steps
          spk_out, mem_out = net(data[step])
          spk_rec.append(spk_out)
      
      return torch.stack(spk_rec)

2.2 Training
^^^^^^^^^^^^

In the previous tutorial, Cross Entropy Loss was applied to the total
spike count to maximize the number of spikes from the correct class.

Another option from the ``snn.functional`` module is to specify the
target number of spikes from correct and incorrect classes. The approach
below uses the *Mean Square Error Spike Count Loss*, which aims to
elicit spikes from the correct class 80% of the time, and 20% of the
time from incorrect classes. Encouraging incorrect neurons to fire could
be motivated to avoid dead neurons.

::

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

Training neuromorphic data is expensive as it requires sequentially
iterating through many time steps (approximately 300 time steps in the
N-MNIST dataset). The following simulation will take some time, so we
will just stick to training across 50 iterations (which is roughly
1/10th of a full epoch). Feel free to change ``num_iters`` if you have
more time to kill. As we are printing results at each iteration, the
results will be quite noisy and will also take some time before we start
to see any sort of improvement.

In our own experiments, it took about 20 iterations before we saw any
improvement, and after 50 iterations, managed to crack ~60% accuracy.

   Warning: the following simulation will take a while. Go make yourself
   a coffee, or ten.

::

    num_epochs = 1
    num_iters = 50
    
    loss_hist = []
    acc_hist = []
    
    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)
    
            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)
    
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
    
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
     
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
    
            acc = SF.accuracy_rate(spk_rec, targets) 
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

            # training loop breaks after 50 iterations
            if i == num_iters:
              break

The output should look something like this:

::

    Epoch 0, Iteration 0 
    Train Loss: 31.00
    Accuracy: 10.16%

    Epoch 0, Iteration 1 
    Train Loss: 30.58
    Accuracy: 13.28%

And after some more time:

::

    Epoch 0, Iteration 49 
    Train Loss: 8.78
    Accuracy: 47.66%

    Epoch 0, Iteration 50 
    Train Loss: 8.43
    Accuracy: 56.25%


3. Results
^^^^^^^^^^

3.1 Plot Test Accuracy
^^^^^^^^^^^^^^^^^^^^^^

::

    import matplotlib.pyplot as plt
    
    # Plot Loss
    fig = plt.figure(facecolor="w")
    plt.plot(acc_hist)
    plt.title("Train Set Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial7/train_acc.png?raw=true
        :align: center
        :width: 450


3.2 Spike Counter
^^^^^^^^^^^^^^^^^

Run a forward pass on a batch of data to obtain spike recordings.

::

    spk_rec = forward_pass(net, data)

Changing ``idx`` allows you to index into various samples from the
simulated minibatch. Use ``splt.spike_count`` to explore the spiking
behaviour of a few different samples. Generating the following animation
will take some time.

   Note: if you are running the notebook locally on your desktop, please
   uncomment the line below and modify the path to your ffmpeg.exe

::

    from IPython.display import HTML
    
    idx = 0
    
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
    print(f"The target label is: {targets[idx]}")
    
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'
    
    #  Plot spike count histogram
    anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels, 
                            animate=True, interpolate=1)
    
    HTML(anim.to_html5_video())
    # anim.save("spike_bar.mp4")

::
^^
    The target label is: 3

.. raw:: html

    <center>
        <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial7/spike_counter.mp4?raw=true"></video>
    </center>

Conclusion
^^^^^^^^^^

If you made it this far, then congratulations - you have the patience of
a monk. You should now also understand how to load neuromorphic datasets
using Tonic and then train a network using snnTorch. 

This concludes the deep-dive tutorial series. 
Check out the `advanced tutorials <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__ 
to learn more advanced techniques, such as introducing long-term temporal dynamics into our SNNs, 
population coding, or accelerating on Intelligence Processing Units.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.

Additional Resources
^^^^^^^^^^^^^^^^^^^^

-  `Check out the snnTorch GitHub project
   here. <https://github.com/jeshraghian/snntorch>`__
-  `The Tonic GitHub project can be found
   here. <https://github.com/neuromorphs/tonic>`__
-  The N-MNIST Dataset was originally published in the following paper:
   `Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N. “Converting
   Static Image Datasets to Spiking Neuromorphic Datasets Using
   Saccades”, Frontiers in Neuroscience, vol.9, no.437,
   Oct. 2015. <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>`__
-  For further information about how N-MNIST was created, please refer
   to `Garrick Orchard’s website
   here. <https://www.garrickorchard.com/datasets/n-mnist>`__


================================================================================
Exoplanet Hunter: Finding Planets Using Light Intensity
================================================================================

Tutorial written by Ruhai Lin, Aled dela Cruz, and Karina Aguilar


.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_exoplanet_hunter.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_exoplanet_hunter.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_



In this tutorial, you will learn:

*  How to train spiking neural networks for time series data,
*  Using SMOTE to deal with unbalanced datasets,
*  Metrics beyond accuracy to evaluate model performance,
*  Some astronomy knowledge

First install the snntorch library before you run any of the code below.

Install the latest PyPi distribution of snnTorch:

::

    pip install snntorch


And then import the libraries as shown in the code below.

::

    # imports
    import snntorch as snn
    from snntorch import surrogate
    
    # pytorch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    
    # SMOTE
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    
    # plot
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # metric (AUC, ROC, sensitivity & specificity)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score

0. Exoplanet Detection (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Before diving into the code, let's gain an understanding of what Exoplanet Detection is.

0.1 Transit Method
^^^^^^^^^^^^^^^^^^

The transit method is a widely used and successful technique for
detecting exoplanets. When an exoplanet transits its host star, it
causes a temporary reduction in the star's light flux (brightness). 
Compared to other techniques, the transit method has discovered 
the largest number of planets.

Astronomers use telescopes equipped with photometers or
spectrophotometers to continuously monitor the brightness of a star over
time. Repeated observations of multiple transits allow astronomers to
gather more detailed information about the exoplanet, such as its
atmosphere and the presence of moons.

Space telescopes like NASA's Kepler and TESS (Transiting Exoplanet
Survey Satellite) have been instrumental in discovering thousands of
exoplanets using the transit method. Without the Earth's atmosphere to hinder observations,
there is minimal interference, allowing for more precise measurements. 
The transit method remains a key tool in furthering our comprehension of
exoplanetary systems. For more information about transit method, you can
visit `NASA Exoplanet Exploration
Page <https://exoplanets.nasa.gov/alien-worlds/ways-to-find-a-planet/#/2>`__.

0.2 Challenges
^^^^^^^^^^^^^^

The drawback of this method is that the angle between the planet's
orbital plane and the direction of the observer's line of sight must be
sufficiently small. Therefore, the chance of this phenomenon occurring is not
high. Thus, more time and resources must be allocated to detect and confirm
the existence of an exoplanet. These resources include the Kepler
telescope and ESA's CoRoT when they were still operational.

Another aspect to consider is power consumption for the device sent
into deep space. For example, a satellite sent into space to observe
light intensity of stars in the solar system. Since the device is in
space, power becomes a limited and valuable resource. Typical AI models
are not suitable for taking the observed data and identifying exoplanets
because of the large amount of energy required to maintain them.

Therefore, a Spiking Neural Network (SNN) could be well suited for this task.

1. Exoplanet Dataset Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following instructions will describe how to obtain the dataset to be used for the SNN.

1.1 Google Drive / Kaggle API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple way to connect the `.csv` file with Google Colab is to put the files in Google Drive. To import our training set and our test set, we need the following two files to be placed in GDrive:

* `'exoTrain.csv'` 
* `'exoTest.csv'` 

They can be downloaded from `Kaggle <https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data>`__.
You can either modify the code blocks below to direct to your selected folder, or create a folder named `SNN`. Put `'exoTrain.csv'` and `'exoTest.csv'` into the folder, and run the code below.

::

    from google.colab import drive
    drive.mount('/content/drive')

::

    cd "/content/drive/My-Drive/SNN"


Use `ls`` to confirm `exoTest.csv` and `exoTrain.csv` are accessible.

.. code:: python

    ls


.. parsed-literal::

    exoTest.csv   exoTrain.csv   SNN_Exoplanet_Hunter_Tutorial.ipynb
    

1.2 Grab the dataset
^^^^^^^^^^^^^^^^^^^^

The code block below is based on the `official PyTorch tutorial on
custom
datasets <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`__

.. code:: python

    # Step 1: Prepare the dataset
    
    class CustomDataset(Dataset):
        def __init__(self, csv_file, transform=None):
            with open(csv_file,"r") as f:
                self.data = pd.read_csv(f) # read the files
            self.labels = self.data.iloc[:,0].values - 1 # set the first line of the input data as the label (Originally 1 or 2, but we -1 here so they become 0 or 1)
            self.features = self.data.iloc[:, 1:].values # set the rest of the input data as the feature (FLUX over time)
            self.transform = transform # transformation (which is None) that will be applied to samples.
    
            # If you want to have a look at how does this dataset look like with pandas,
            # you can enable the line below.
            # print(data.head(5))
    
        def __len__(self): # function that gives back the size of the dataset (how many samples)
            return len(self.labels)
    
        def __getitem__(self, idx): # retrieves a data sample from the dataset
            label = self.labels[idx] # fetch label of sample
            feature = self.features[idx] # fetch features of sample
    
            if self.transform: # if there is a specified transformation, transform the data
                feature = self.transform(feature)
    
            sample = {'feature': feature, 'label': label}
            return sample
    
    train_dataset = CustomDataset('./exoTrain.csv') # grab the training data
    test_dataset = CustomDataset('./exoTest.csv') # grab the test data
    # print(train_dataset.__getitem__(37));

1.3 Augmenting the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the low chance of detecting exoplanets, this dataset is very imbalanced.
Most samples are negative, meaning there are very few exoplanets from the observed
light intensity data. If your model were to simply predict 'no exoplanet' for every sample,
then it would achieve very high accuracy. This indicates that accuracy is a poor metric for success.

Let's first probe our data to gain insight into how imbalanced it is.

.. code:: python

    print("Class distribution in the original training dataset:", pd.Series(train_dataset.labels).value_counts())
    print("Class distribution in the original testing dataset:", pd.Series(test_dataset.labels).value_counts())


.. parsed-literal::

    Class distribution in the original training dataset: 0    5050
    1      37
    dtype: int64
    Class distribution in the original testing dataset: 0    565
    1      5
    dtype: int64
    
I.e., there are 5050 negative samples and only 37 positive samples in the training set. 


.. code:: python

    label_counts = np.bincount(train_dataset.labels)
    label_names = ['Not Exoplanet','Exoplanet']
    
    plt.figure(figsize=(8, 6))
    plt.pie(label_counts, labels=label_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title('Distribution of Positive and Negative Samples in the Training Dataset')
    plt.show()



.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_exoplanet_hunter_files/tutorial_exoplanet_hunter_24_0.png?raw=true 
    

To deal with the imbalance of our dataset, let's Synthetic Minority
Over-Sampling Technique (SMOTE). SMOTE works by
generating synthetic samples from the minority class to balance the
distribution (typically implemented using the nearest neighbors
strategy). By implementing SMOTE, we attempt to reduce bias toward
stars without exoplanets (the majority class).

.. code:: python

    # Step 2: Apply SMOTE to deal with the unbalanced data
    smote = SMOTE(sampling_strategy='all') # initialize a smote, while sampling_strategy='all' means setting all the classes to the same size
    train_dataset.features, train_dataset.labels = smote.fit_resample(train_dataset.features, train_dataset.labels) # update the labels and features to the resampled data
    
    print("Class distribution in the training dataset after SMOTE:", pd.Series(train_dataset.labels).value_counts())
    print("Class distribution in the testing dataset after SMOTE:", pd.Series(test_dataset.labels).value_counts())


.. parsed-literal::

    Class distribution in the training dataset after SMOTE: 1    5050
    0    5050
    dtype: int64
    Class distribution in the testing dataset after SMOTE: 0    565
    1      5
    dtype: int64
    

.. code:: python

    label_counts = np.bincount(train_dataset.labels)
    label_names = ['Not Exoplanet','Exoplanet']
    
    plt.figure(figsize=(8, 6))
    plt.pie(label_counts, labels=label_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title('Distribution of Positive and Negative Samples')
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_exoplanet_hunter_files/tutorial_exoplanet_hunter_27_0.png?raw=true 



1.4 Create the DataLoader
^^^^^^^^^^^^^^^^^^^^^^^^^

We will create a dataloader to help batch and shuffle the data during training
and testing. In the initialization of the dataloader, the parameters
include: the dataset to be loaded, the batch size, a shuffle argument to
determine whether or not to shuffle the dataset after each epoch, and a
`drop_last` parameter that decides whether or not a potential final
“incomplete” batch is dropped.

.. code:: python

    # Step 3: Create dataloader
    batch_size = 64 # determines the number of samples in each batch during training
    spike_grad = surrogate.fast_sigmoid(slope=25) #
    beta = 0.5 # initialize a beta value of 0.5
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # create a dataloader for the trainset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True) # create a dataloader for the testset

1.5 Description of the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After loading the data, let's see what our data looks like.

.. code:: python

    print(train_dataset.data.head(1))


.. parsed-literal::

       LABEL  FLUX.1  FLUX.2  FLUX.3  FLUX.4  FLUX.5  FLUX.6  FLUX.7  FLUX.8  \
    0      2   93.85   83.81    20.1  -26.98  -39.56 -124.71 -135.18  -96.27   
    
       FLUX.9  ...  FLUX.3188  FLUX.3189  FLUX.3190  FLUX.3191  FLUX.3192  \
    0  -79.89  ...     -78.07    -102.15    -102.15      25.13      48.57   
    
       FLUX.3193  FLUX.3194  FLUX.3195  FLUX.3196  FLUX.3197  
    0      92.54      39.32      61.42       5.08     -39.54  
    
    [1 rows x 3198 columns]
    


.. code:: python

    fig = make_subplots(rows=2, cols=2,subplot_titles=("Star #0 (Exoplanet)", "Star #1 (Exoplanet)",
                                                       "Star #3000 (No-Exoplanet)", "Star #3001 (No-Exoplanet)"))
    fig.add_trace(
        go.Scatter(y=train_dataset.__getitem__(0)['feature']),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=train_dataset.__getitem__(1)['feature']),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=train_dataset.__getitem__(3000)['feature']),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=train_dataset.__getitem__(3001)['feature']),
        row=2, col=2
    )
    for i in range(1, 5):
        fig.update_xaxes(title_text="Time", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_yaxes(title_text="Flux", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    
    fig.update_layout(height=600, width=800, title_text="Exoplanets Flux vs No-Exoplanet Flux",showlegend=False)
    fig.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_exoplanet_hunter_files/newplot.png?raw=true 




2. Train and Test
^^^^^^^^^^^^^^^^^

2.1 Define Network with snnTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code block below follows the same syntax as with the `official
snnTorch
tutorial <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__. 
In contrast to other tutorials, however, this model concurrently processes data across the entire sequence. 
In that sense, it is more akin to how attention-based mechanisms handle data.
Turning this into a more 'online' method would likely involve preprocessing to downsample the exceedingly long sequence length.

.. code:: python

    # Step 4: Define Network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
    
            # Initialize layers (3 linear layers and 3 leaky layers)
            self.fc1 = nn.Linear(3197, 128) # takes an input of 3197 and outputs 128
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc2 = nn.Linear(64, 64) # takes an input of 64 and outputs 68
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc3 = nn.Linear(32, 2) # takes in 32 inputs and outputs our two outputs (planet with/without an exoplanet)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        def forward(self, x):
    
            # Initialize hidden states and outputs at t=0
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
    
            cur1 = F.max_pool1d(self.fc1(x), 2)
            spk1, mem1 = self.lif1(cur1, mem1)
    
            cur2 = F.max_pool1d(self.fc2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)
    
            cur3 = self.fc3(spk2.view(batch_size, -1))
    
            # return cur3
            return cur3
    
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = Net() # initialize the model to the new class.

2.2 Define the Loss function and the Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Step 5: Define the Loss function and the Optimizer
    criterion = nn.CrossEntropyLoss()  # look up binarycross entropy if we have time
    optimizer = optim.SGD(model.parameters(), lr=0.001) # stochastic gradient descent with a learning rate of 0.001

2.3 Train and Test the Model over each EPOCH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sensitivity
^^^^^^^^^^^

Sensitivity (Recall / True Positive Rate) measures the proportion of
actual positive cases correctly identified by the model. It indicates
the model's ability to correctly detect or capture all positive
instances out of the total actual positives.

.. math:: Sensitivity =  \frac{TP}{TP+FN} \tag{1}

TP stands for the True Positive prediction, which means the number of
the positive samples that are correctly predicted. FN stands for the
False Negative prediction, which means the number of the negative
samples that are mistakenly predicted as positive samples.

Specificity
^^^^^^^^^^^

On the other hand, specificity measures the proportion of actual
negative cases correctly identified by the model. It indicates the
model's ability to correctly exclude negative instances out of the total
actual negatives.

.. math:: Specificity =  \frac{TN}{TN+FP} \tag{2}

Similarly, TN stands for the True Negative prediction, FP stands for the
False Positive prediction.

AUC-ROC
^^^^^^^

The AUC-ROC (Area Under the Receiver Operating Characteristic curve)
metric is commonly used for evaluating the performance of binary
classification models, plotting the true positive rate against the false
positive rate. It quantifies the model's ability to distinguish between
classes, specifically its capacity to correctly rank or order predicted
probabilities.

roc_auc_score(): returns a value between 0 or 1.

* Values :math:`> 0.5` and closer to 1 indicate that the model does well in distinguishing between the two classes 
* Values close to 0.5 represent that the model does no better than random guessing 
* Values :math:`< 0.5`` demonstrate that the model performs worse than random guessing

Since there are minimal test values for stars with exoplanets, these
metrics are far better than accuracy alone for determining model performance. Let's list
all the varaiables that we need:

.. code:: python

    # create a pandas dataframe to hold the current epoch, the accuracy， sensitivity, specificity, auc-roc and loss
    results = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC-ROC', 'Test Loss'])

And then define how many epochs we want the model to be trained:

.. code:: python

    num_epochs = 100 # initialize a certain number of epoch iterations

Note that the best range for epochs is around 50 to 500 for our dataset.

Now let's train the model.

.. code:: python

    for epoch in range(num_epochs): # iterate through num_epochs
        model.train() # forward pass
        for data in train_dataloader: # iterate through every data sample
            inputs, labels = data['feature'].float(), data['label']  # Float
            optimizer.zero_grad() # clear previously stored gradients
            outputs = model(inputs) #
            loss = criterion(outputs, labels) # calculates the difference (loss) between actual values and predictions
            loss.backward() # backward pass on the loss
            optimizer.step() # updates parameters
    
        # Test Set, evaluate the model every epoch
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_predicted = []
            all_probs = []
            for data in test_dataloader:
                inputs, labels = data['feature'].float(), data['label']
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
    
^^^^
                softmax = torch.nn.Softmax(dim=1)
                probabilities = softmax(outputs)[:, 1]  # Assuming 1 represents the positive class
                all_probs.extend(probabilities.cpu().numpy())
            # output the accuracy (even though it is not very useful in this case)
            accuracy = 100 * correct / total
            # calculate teat loss
            # test_loss =
            # initialize a confusing matrix
            cm = confusion_matrix(all_labels, all_predicted)
            # grab the amount of true negatives and positives, and false negatives and positives.
            tn, fp, fn, tp = cm.ravel()
            # calculate sensitivity
            sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # calculate specificity
            specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0.0
            # calculate AUC-ROC
            auc_roc = 100 * roc_auc_score(all_labels, all_probs)
            print(
                f'Epoch [{epoch + 1}/{num_epochs}] Test Loss: {test_loss / len(test_dataloader):.2f} '
                f'Test Accuracy: {accuracy:.2f}% Sensitivity: {sensitivity:.2f}% Specificity: {specificity:.2f}% AUC-ROC: {auc_roc:.4f}%'
            )
    
            results = results._append({
                'Epoch': epoch + 1,
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Test Loss': test_loss / len(test_dataloader),
                'AUC-ROC': auc_roc
            }, ignore_index=True)


.. parsed-literal::

    Epoch [1/100] Test Loss: 0.68 Test Accuracy: 67.38% Sensitivity: 80.00% Specificity: 67.26% AUC-ROC: 72.6824%
    Epoch [2/100] Test Loss: 0.69 Test Accuracy: 66.99% Sensitivity: 80.00% Specificity: 66.86% AUC-ROC: 72.5247%
    Epoch [3/100] Test Loss: 0.69 Test Accuracy: 59.77% Sensitivity: 80.00% Specificity: 59.57% AUC-ROC: 72.4063%
    Epoch [4/100] Test Loss: 0.70 Test Accuracy: 59.18% Sensitivity: 80.00% Specificity: 58.97% AUC-ROC: 72.0316%
    Epoch [5/100] Test Loss: 0.70 Test Accuracy: 58.59% Sensitivity: 80.00% Specificity: 58.38% AUC-ROC: 71.4596%
    Epoch [6/100] Test Loss: 0.70 Test Accuracy: 58.01% Sensitivity: 80.00% Specificity: 57.79% AUC-ROC: 71.2032%
    Epoch [7/100] Test Loss: 0.71 Test Accuracy: 57.23% Sensitivity: 80.00% Specificity: 57.00% AUC-ROC: 70.6114%
    Epoch [8/100] Test Loss: 0.71 Test Accuracy: 57.03% Sensitivity: 80.00% Specificity: 56.80% AUC-ROC: 70.4339%
    Epoch [9/100] Test Loss: 0.71 Test Accuracy: 57.03% Sensitivity: 80.00% Specificity: 56.80% AUC-ROC: 70.2761%
    Epoch [10/100] Test Loss: 0.71 Test Accuracy: 56.84% Sensitivity: 80.00% Specificity: 56.61% AUC-ROC: 70.1183%
    Epoch [11/100] Test Loss: 0.71 Test Accuracy: 28.32% Sensitivity: 100.00% Specificity: 27.61% AUC-ROC: 71.8738%
    Epoch [12/100] Test Loss: 0.71 Test Accuracy: 28.32% Sensitivity: 100.00% Specificity: 27.61% AUC-ROC: 74.9704%
    Epoch [13/100] Test Loss: 0.71 Test Accuracy: 28.32% Sensitivity: 100.00% Specificity: 27.61% AUC-ROC: 76.6667%
    Epoch [14/100] Test Loss: 0.71 Test Accuracy: 34.57% Sensitivity: 100.00% Specificity: 33.93% AUC-ROC: 73.9448%
    Epoch [15/100] Test Loss: 0.71 Test Accuracy: 34.38% Sensitivity: 100.00% Specificity: 33.73% AUC-ROC: 73.8659%
    Epoch [16/100] Test Loss: 0.71 Test Accuracy: 34.57% Sensitivity: 100.00% Specificity: 33.93% AUC-ROC: 73.5108%
    Epoch [17/100] Test Loss: 0.71 Test Accuracy: 34.77% Sensitivity: 100.00% Specificity: 34.12% AUC-ROC: 78.5010%
    Epoch [18/100] Test Loss: 0.71 Test Accuracy: 59.57% Sensitivity: 80.00% Specificity: 59.37% AUC-ROC: 78.1854%
    Epoch [19/100] Test Loss: 0.70 Test Accuracy: 59.38% Sensitivity: 80.00% Specificity: 59.17% AUC-ROC: 77.9684%
    Epoch [20/100] Test Loss: 0.70 Test Accuracy: 59.18% Sensitivity: 80.00% Specificity: 58.97% AUC-ROC: 77.5148%
    Epoch [21/100] Test Loss: 0.70 Test Accuracy: 58.40% Sensitivity: 80.00% Specificity: 58.19% AUC-ROC: 77.1795%
    Epoch [22/100] Test Loss: 0.70 Test Accuracy: 58.01% Sensitivity: 80.00% Specificity: 57.79% AUC-ROC: 77.1203%
    Epoch [23/100] Test Loss: 0.70 Test Accuracy: 56.84% Sensitivity: 80.00% Specificity: 56.61% AUC-ROC: 76.4892%
    Epoch [24/100] Test Loss: 0.70 Test Accuracy: 56.05% Sensitivity: 80.00% Specificity: 55.82% AUC-ROC: 76.2130%
    Epoch [25/100] Test Loss: 0.70 Test Accuracy: 54.49% Sensitivity: 80.00% Specificity: 54.24% AUC-ROC: 75.5819%
    Epoch [26/100] Test Loss: 0.70 Test Accuracy: 54.49% Sensitivity: 80.00% Specificity: 54.24% AUC-ROC: 75.6607%
    Epoch [27/100] Test Loss: 0.70 Test Accuracy: 55.27% Sensitivity: 80.00% Specificity: 55.03% AUC-ROC: 75.4438%
    Epoch [28/100] Test Loss: 0.69 Test Accuracy: 55.27% Sensitivity: 80.00% Specificity: 55.03% AUC-ROC: 75.4832%
    Epoch [29/100] Test Loss: 0.69 Test Accuracy: 54.69% Sensitivity: 80.00% Specificity: 54.44% AUC-ROC: 75.4043%
    Epoch [30/100] Test Loss: 0.69 Test Accuracy: 54.10% Sensitivity: 80.00% Specificity: 53.85% AUC-ROC: 75.1874%
    Epoch [31/100] Test Loss: 0.69 Test Accuracy: 54.10% Sensitivity: 80.00% Specificity: 53.85% AUC-ROC: 75.1085%
    Epoch [32/100] Test Loss: 0.68 Test Accuracy: 53.71% Sensitivity: 80.00% Specificity: 53.45% AUC-ROC: 74.8126%
    Epoch [33/100] Test Loss: 0.68 Test Accuracy: 52.93% Sensitivity: 80.00% Specificity: 52.66% AUC-ROC: 74.3590%
    Epoch [34/100] Test Loss: 0.68 Test Accuracy: 53.32% Sensitivity: 80.00% Specificity: 53.06% AUC-ROC: 74.7337%
    Epoch [35/100] Test Loss: 0.68 Test Accuracy: 53.12% Sensitivity: 80.00% Specificity: 52.86% AUC-ROC: 74.8323%
    Epoch [36/100] Test Loss: 0.68 Test Accuracy: 52.34% Sensitivity: 80.00% Specificity: 52.07% AUC-ROC: 74.4379%
    Epoch [37/100] Test Loss: 0.67 Test Accuracy: 53.71% Sensitivity: 80.00% Specificity: 53.45% AUC-ROC: 74.8521%
    Epoch [38/100] Test Loss: 0.67 Test Accuracy: 53.71% Sensitivity: 80.00% Specificity: 53.45% AUC-ROC: 75.0493%
    Epoch [39/100] Test Loss: 0.67 Test Accuracy: 53.52% Sensitivity: 80.00% Specificity: 53.25% AUC-ROC: 74.9507%
    Epoch [40/100] Test Loss: 0.66 Test Accuracy: 53.91% Sensitivity: 80.00% Specificity: 53.65% AUC-ROC: 75.2465%
    Epoch [41/100] Test Loss: 0.66 Test Accuracy: 54.10% Sensitivity: 80.00% Specificity: 53.85% AUC-ROC: 75.5424%
    Epoch [42/100] Test Loss: 0.65 Test Accuracy: 54.69% Sensitivity: 80.00% Specificity: 54.44% AUC-ROC: 75.8580%
    Epoch [43/100] Test Loss: 0.65 Test Accuracy: 66.21% Sensitivity: 80.00% Specificity: 66.07% AUC-ROC: 75.6607%
    Epoch [44/100] Test Loss: 0.65 Test Accuracy: 66.41% Sensitivity: 80.00% Specificity: 66.27% AUC-ROC: 76.2130%
    Epoch [45/100] Test Loss: 0.64 Test Accuracy: 67.19% Sensitivity: 80.00% Specificity: 67.06% AUC-ROC: 76.8245%
    Epoch [46/100] Test Loss: 0.64 Test Accuracy: 67.58% Sensitivity: 80.00% Specificity: 67.46% AUC-ROC: 77.0611%
    Epoch [47/100] Test Loss: 0.63 Test Accuracy: 68.16% Sensitivity: 80.00% Specificity: 68.05% AUC-ROC: 77.1992%
    Epoch [48/100] Test Loss: 0.63 Test Accuracy: 68.36% Sensitivity: 80.00% Specificity: 68.24% AUC-ROC: 77.2781%
    Epoch [49/100] Test Loss: 0.63 Test Accuracy: 68.55% Sensitivity: 80.00% Specificity: 68.44% AUC-ROC: 78.9941%
    Epoch [50/100] Test Loss: 0.62 Test Accuracy: 69.14% Sensitivity: 80.00% Specificity: 69.03% AUC-ROC: 79.0138%
    Epoch [51/100] Test Loss: 0.62 Test Accuracy: 69.73% Sensitivity: 80.00% Specificity: 69.63% AUC-ROC: 79.4872%
    Epoch [52/100] Test Loss: 0.61 Test Accuracy: 70.12% Sensitivity: 80.00% Specificity: 70.02% AUC-ROC: 79.8028%
    Epoch [53/100] Test Loss: 0.61 Test Accuracy: 71.09% Sensitivity: 80.00% Specificity: 71.01% AUC-ROC: 80.4536%
    Epoch [54/100] Test Loss: 0.60 Test Accuracy: 71.88% Sensitivity: 80.00% Specificity: 71.79% AUC-ROC: 80.9467%
    Epoch [55/100] Test Loss: 0.60 Test Accuracy: 72.07% Sensitivity: 80.00% Specificity: 71.99% AUC-ROC: 80.9467%
    Epoch [56/100] Test Loss: 0.59 Test Accuracy: 72.27% Sensitivity: 80.00% Specificity: 72.19% AUC-ROC: 80.8087%
    Epoch [57/100] Test Loss: 0.59 Test Accuracy: 73.44% Sensitivity: 80.00% Specificity: 73.37% AUC-ROC: 81.2032%
    Epoch [58/100] Test Loss: 0.59 Test Accuracy: 74.02% Sensitivity: 80.00% Specificity: 73.96% AUC-ROC: 81.3412%
    Epoch [59/100] Test Loss: 0.58 Test Accuracy: 75.00% Sensitivity: 80.00% Specificity: 74.95% AUC-ROC: 81.5779%
    Epoch [60/100] Test Loss: 0.58 Test Accuracy: 75.00% Sensitivity: 80.00% Specificity: 74.95% AUC-ROC: 80.1775%
    Epoch [61/100] Test Loss: 0.57 Test Accuracy: 75.39% Sensitivity: 80.00% Specificity: 75.35% AUC-ROC: 80.0000%
    Epoch [62/100] Test Loss: 0.57 Test Accuracy: 76.95% Sensitivity: 80.00% Specificity: 76.92% AUC-ROC: 80.6509%
    Epoch [63/100] Test Loss: 0.56 Test Accuracy: 77.54% Sensitivity: 80.00% Specificity: 77.51% AUC-ROC: 81.0256%
    Epoch [64/100] Test Loss: 0.55 Test Accuracy: 78.52% Sensitivity: 80.00% Specificity: 78.50% AUC-ROC: 81.8540%
    Epoch [65/100] Test Loss: 0.55 Test Accuracy: 78.52% Sensitivity: 80.00% Specificity: 78.50% AUC-ROC: 81.4990%
    Epoch [66/100] Test Loss: 0.54 Test Accuracy: 78.71% Sensitivity: 80.00% Specificity: 78.70% AUC-ROC: 81.7357%
    Epoch [67/100] Test Loss: 0.54 Test Accuracy: 77.93% Sensitivity: 80.00% Specificity: 77.91% AUC-ROC: 77.6726%
    Epoch [68/100] Test Loss: 0.53 Test Accuracy: 78.91% Sensitivity: 80.00% Specificity: 78.90% AUC-ROC: 78.2249%
    Epoch [69/100] Test Loss: 0.53 Test Accuracy: 78.71% Sensitivity: 80.00% Specificity: 78.70% AUC-ROC: 78.3629%
    Epoch [70/100] Test Loss: 0.52 Test Accuracy: 79.49% Sensitivity: 80.00% Specificity: 79.49% AUC-ROC: 78.9349%
    Epoch [71/100] Test Loss: 0.51 Test Accuracy: 80.08% Sensitivity: 80.00% Specificity: 80.08% AUC-ROC: 79.4280%
    Epoch [72/100] Test Loss: 0.51 Test Accuracy: 80.66% Sensitivity: 80.00% Specificity: 80.67% AUC-ROC: 79.5464%
    Epoch [73/100] Test Loss: 0.51 Test Accuracy: 80.86% Sensitivity: 80.00% Specificity: 80.87% AUC-ROC: 79.7239%
    Epoch [74/100] Test Loss: 0.50 Test Accuracy: 81.64% Sensitivity: 80.00% Specificity: 81.66% AUC-ROC: 79.8422%
    Epoch [75/100] Test Loss: 0.49 Test Accuracy: 82.03% Sensitivity: 80.00% Specificity: 82.05% AUC-ROC: 80.2367%
    Epoch [76/100] Test Loss: 0.48 Test Accuracy: 82.23% Sensitivity: 60.00% Specificity: 82.45% AUC-ROC: 75.8580%
    Epoch [77/100] Test Loss: 0.48 Test Accuracy: 82.42% Sensitivity: 60.00% Specificity: 82.64% AUC-ROC: 76.3116%
    Epoch [78/100] Test Loss: 0.47 Test Accuracy: 82.62% Sensitivity: 60.00% Specificity: 82.84% AUC-ROC: 76.7258%
    Epoch [79/100] Test Loss: 0.46 Test Accuracy: 83.20% Sensitivity: 60.00% Specificity: 83.43% AUC-ROC: 77.4162%
    Epoch [80/100] Test Loss: 0.46 Test Accuracy: 82.81% Sensitivity: 60.00% Specificity: 83.04% AUC-ROC: 77.7515%
    Epoch [81/100] Test Loss: 0.45 Test Accuracy: 83.01% Sensitivity: 60.00% Specificity: 83.23% AUC-ROC: 78.0671%
    Epoch [82/100] Test Loss: 0.45 Test Accuracy: 82.81% Sensitivity: 60.00% Specificity: 83.04% AUC-ROC: 77.8304%
    Epoch [83/100] Test Loss: 0.44 Test Accuracy: 83.20% Sensitivity: 60.00% Specificity: 83.43% AUC-ROC: 78.4221%
    Epoch [84/100] Test Loss: 0.43 Test Accuracy: 83.98% Sensitivity: 60.00% Specificity: 84.22% AUC-ROC: 78.6391%
    Epoch [85/100] Test Loss: 0.43 Test Accuracy: 84.18% Sensitivity: 60.00% Specificity: 84.42% AUC-ROC: 78.9744%
    Epoch [86/100] Test Loss: 0.43 Test Accuracy: 83.98% Sensitivity: 60.00% Specificity: 84.22% AUC-ROC: 78.9546%
    Epoch [87/100] Test Loss: 0.42 Test Accuracy: 84.18% Sensitivity: 60.00% Specificity: 84.42% AUC-ROC: 79.0335%
    Epoch [88/100] Test Loss: 0.42 Test Accuracy: 84.38% Sensitivity: 60.00% Specificity: 84.62% AUC-ROC: 79.1913%
    Epoch [89/100] Test Loss: 0.41 Test Accuracy: 85.74% Sensitivity: 40.00% Specificity: 86.19% AUC-ROC: 74.6351%
    Epoch [90/100] Test Loss: 0.40 Test Accuracy: 86.13% Sensitivity: 40.00% Specificity: 86.59% AUC-ROC: 67.5740%
    Epoch [91/100] Test Loss: 0.40 Test Accuracy: 86.72% Sensitivity: 40.00% Specificity: 87.18% AUC-ROC: 67.8107%
    Epoch [92/100] Test Loss: 0.40 Test Accuracy: 86.33% Sensitivity: 40.00% Specificity: 86.79% AUC-ROC: 67.8304%
    Epoch [93/100] Test Loss: 0.39 Test Accuracy: 86.13% Sensitivity: 40.00% Specificity: 86.59% AUC-ROC: 66.8639%
    Epoch [94/100] Test Loss: 0.38 Test Accuracy: 86.33% Sensitivity: 40.00% Specificity: 86.79% AUC-ROC: 67.1795%
    Epoch [95/100] Test Loss: 0.39 Test Accuracy: 86.52% Sensitivity: 40.00% Specificity: 86.98% AUC-ROC: 67.0809%
    Epoch [96/100] Test Loss: 0.37 Test Accuracy: 86.72% Sensitivity: 40.00% Specificity: 87.18% AUC-ROC: 67.7120%
    Epoch [97/100] Test Loss: 0.37 Test Accuracy: 87.11% Sensitivity: 60.00% Specificity: 87.38% AUC-ROC: 73.4911%
    Epoch [98/100] Test Loss: 0.37 Test Accuracy: 87.11% Sensitivity: 60.00% Specificity: 87.38% AUC-ROC: 73.4320%
    Epoch [99/100] Test Loss: 0.36 Test Accuracy: 87.70% Sensitivity: 60.00% Specificity: 87.97% AUC-ROC: 73.6292%
    Epoch [100/100] Test Loss: 0.36 Test Accuracy: 87.70% Sensitivity: 60.00% Specificity: 87.97% AUC-ROC: 73.5897%

The process may be finnicky. Better specificity usually comes at the cost of sensitivity. 
In our case, we generally see good results anywhere between 50-500 epochs depending on the seed. 
Too many epochs, and the model tends to overfit to the negative samples. 

3. Visualize the Results
^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_exoplanet_hunter_files/acc.png?raw=true 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_exoplanet_hunter_files/sensitivity.png?raw=true 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_exoplanet_hunter_files/specificity.png?raw=true 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_exoplanet_hunter_files/aucroc.png?raw=true 


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_exoplanet_hunter_files/all-results.png?raw=true 




Conlusion
^^^^^^^^^

.. code:: python

    # Save the model if needed
    # torch.save(model.state_dict(), 'custom_model.pth')

You should now have a grasp of how spiking neural network can contribute
to deep space mission along with some limitations.

A special thanks to Giridhar Vadhul and Sahil Konjarla for their super helpful
advice on defining and training the network.

If you like this project, please consider starring ⭐ the snnTorch repo on GitHub
as it is the easiest and best way to support it.


===================================================
Accelerating snnTorch on IPUs
===================================================


Tutorial written by `Jason K. Eshraghian <https://www.jasoneshraghian.com>`_ and Vincent Sun

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. An editable script is available via the following link:
    * `Python Script (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples/tutorial_ipu_1.py>`_


Introduction
^^^^^^^^^^^^

Spiking neural networks (SNNs) have achieved orders of magnitude improvement in terms of energy consumption and latency when performing inference with deep learning workloads.
But in a twist of irony, using error backpropagation to train SNNs becomes more expensive than non-spiking network when trained on CPUs and GPUs.
The additional temporal dimension must be accounted for, and memory complexity increases lineary with time when a network is trained using the backpropagation-through-time algorithm.

An alternative build of snnTorch has been optimized for `Graphcore's Intelligence Processing Units (IPUs) <https://www.graphcore.ai/>`_.
IPUs are custom accelerators tailored for deep learning workloads, and adopt multi-instruction multi-data (MIMD) parallelism by running individual processing threads on smaller blocks of data.
This is an ideal fit for partitions of spiking neuron dynamical state equations that must be sequentially processed, and cannot be vectorized.


In this tutorial, you will: 

    * Learn how to train a SNN accelerated using IPUs.


Ensure up-to-date versions of :code:`poptorch` and the Poplar SDK are installed. Refer to `Graphcore's documentation <https://github.com/graphcore/poptorch>`_ for installation instructions.

Install :code:`snntorch-ipu` in an environment that does not have :code:`snntorch` pre-installed to avoid package conflicts:

::

    !pip install snntorch-ipu

Import the required Python packages:

::

    import torch, torch.nn as nn
    import popart, poptorch
    import snntorch as snn
    import snntorch.functional as SF

DataLoading
^^^^^^^^^^^

Load in the MNIST dataset.

::

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    batch_size = 128
    data_path='/tmp/data/mnist'
    
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    # Train using full precision 32-flt
    opts = poptorch.Options()
    opts.Precision.halfFloatCasting(poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)

    # Create DataLoaders
    train_loader = poptorch.DataLoader(options=opts, dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=20)
    test_loader = poptorch.DataLoader(options=opts, dataset=mnist_test, batch_size=batch_size, shuffle=True, num_workers=20)


Define Network
^^^^^^^^^^^^^^

Let's simulate our network for 25 time steps using a slow state-decay rate for our spiking neurons:

::

    num_steps = 25
    beta = 0.9


We will now construct a vanilla SNN model. 
When training on IPUs, note that the loss function must be wrapped within the model class.
The full code will look this:

::

    class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 784
        num_hidden = 1000
        num_outputs = 10

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Leaky(beta=beta)

        # Cross-Entropy Spike Count Loss
        self.loss_fn = SF.ce_count_loss()

    def forward(self, x, labels=None):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []
       
        for step in range(num_steps):
            cur1 = self.fc1(x.view(batch_size,-1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        spk2_rec = torch.stack(spk2_rec)
        mem2_rec = torch.stack(mem2_rec)

        if self.training:
            return spk2_rec, poptorch.identity_loss(self.loss_fn(mem2_rec, labels), "none")
        return spk2_rec


Let's quickly break this down. 

Contructing the model is the same as all previous tutorials. We apply spiking neuron nodes at the end of each dense layer:

::

    self.fc1 = nn.Linear(num_inputs, num_hidden)
    self.lif1 = snn.Leaky(beta=beta)
    self.fc2 = nn.Linear(num_hidden, num_output)
    self.lif2 = snn.Leaky(beta=beta)

By default, the surrogate gradient of the spiking neurons will be a straight through estimator.
Fast Sigmoid and Sigmoid options are also available if you prefer to use those:

::

    from snntorch import surrogate

    self.lif1 = snn.Leaky(beta=beta, spike_grad = surrogate.fast_sigmoid())


The loss function will count up the total number of spikes from each output neuron and apply the Cross Entropy Loss:

::

    self.loss_fn = SF.ce_count_loss()

Now we define the forward pass. Initialize the hidden state of each spiking neuron by calling the following functions:

::

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()


Next, run the for-loop to simulate the SNN over 25 time steps.
The input data is flattened using :code:`.view(batch_size, -1)` to make it compatible with a dense input layer.

::

    for step in range(num_steps):
        cur1 = self.fc1(x.view(batch_size,-1))
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

The loss is applied using the function :code:`poptorch.identity_loss(self.loss_fn(mem2_rec, labels), "none")`.


Training on IPUs
^^^^^^^^^^^^^^^^

Now, the full training loop is run across 10 epochs. 
Note the optimizer is called from :code:`poptorch`. Otherwise, the training process is much the same as in typical use of snnTorch.

::

    net = Model()
    optimizer = poptorch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    poptorch_model = poptorch.trainingModel(net, options=opts, optimizer=optimizer)

    epochs = 10
    for epoch in tqdm(range(epochs), desc="epochs"):
        correct = 0.0

        for i, (data, labels) in enumerate(train_loader):
            output, loss = poptorch_model(data, labels)

            if i % 250 == 0:
                _, pred = output.sum(dim=0).max(1)
                correct = (labels == pred).sum().item()/len(labels)

                # Accuracy on a single batch
                print("Accuracy: ", correct)

The model will first be compiled, after which, the training process will commence. 
The accuracy will be printed out for individual minibatches on the training set to keep this tutorial quick and minimal.


Conclusion
^^^^^^^^^^

Our initial benchmarks on show improvements of up to 10x improvements over CUDA accelerated SNNs in mixed-precision training throughput across a variety of neuron models.
A detailed benchmark and blog highlighting additional features are currently under construction.

-  For a detailed tutorial of spiking neurons, neural nets, encoding,
   and training using neuromorphic datasets, check out the `snnTorch
   tutorial
   series <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__.
-  For more information on the features of snnTorch, check out the
   `documentation at this
   link <https://snntorch.readthedocs.io/en/latest/>`__.
-  If you have ideas, suggestions or would like to find ways to get
   involved, then `check out the snnTorch GitHub project
   here. <https://github.com/jeshraghian/snntorch>`__


===================================================
Population Coding in Spiking Neural Nets
===================================================


Tutorial written by Jason K. Eshraghian (`www.jasoneshraghian.com <https://www.jasoneshraghian.com>`_)

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_pop.ipynb

The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_pop.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


Introduction
^^^^^^^^^^^^

It is thought that rate codes alone cannot be the dominant encoding
mechanism in the primary cortex. One of several reasons is because the
average neuronal firing rate is roughly :math:`0.1-1` Hz, which is far
slower than the reaction response time of animals and humans.

But if we pool together multiple neurons and count their spikes
together, then it becomes possible to measure a firing rate for a
population of neurons in a very short window of time. Population coding
adds some credibility to the plausibility of rate-encoding mechanisms.


   .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_pop/pop.png?raw=true
            :align: center
            :width: 300

In this tutorial, you will: 

    * Learn how to train a population coded network. Instead of assigning one neuron per class, we will extend this to multiple neurons per class, and aggregate their spikes together.


::

    !pip install snntorch

::

    import torch, torch.nn as nn
    import snntorch as snn

DataLoading
^^^^^^^^^^^

Define variables for dataloading.

::

    batch_size = 128
    data_path='/tmp/data/fmnist'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

Load FashionMNIST dataset.

::

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    fmnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)

Define Network
^^^^^^^^^^^^^^

Let’s compare the performance of a pair of networks both with and
without population coding, and train them for *one single time step.*

::

    from snntorch import surrogate
    
    # network parameters
    num_inputs = 28*28
    num_hidden = 128
    num_outputs = 10
    num_steps = 1
    
    # spiking neuron parameters
    beta = 0.9  # neuron decay rate 
    grad = surrogate.fast_sigmoid()

Without population coding
^^^^^^^^^^^^^^^^^^^^^^^^^

Let’s just use a simple 2-layer dense spiking network.

::

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hidden),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                        nn.Linear(num_hidden, num_outputs),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                        ).to(device)

With population coding
^^^^^^^^^^^^^^^^^^^^^^

Instead of 10 output neurons corresponding to 10 output classes, we will
use 500 output neurons. This means each output class has 50 neurons
randomly assigned to it.

::

    pop_outputs = 500
    
    net_pop = nn.Sequential(nn.Flatten(),
                            nn.Linear(num_inputs, num_hidden),
                            snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                            nn.Linear(num_hidden, pop_outputs),
                            snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                            ).to(device)

Training
^^^^^^^^

Without population coding
^^^^^^^^^^^^^^^^^^^^^^^^^

Define the optimizer and loss function. Here, we use the MSE Count Loss,
which counts up the total number of output spikes at the end of the
simulation run.

The correct class has a target firing probability of 100%, and incorrect
classes are set to 0%.

::

    import snntorch.functional as SF
    
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0)

We will also define a simple test accuracy function that predicts the
correct class based on the neuron with the highest spike count.

::

    from snntorch import utils
    
    def test_accuracy(data_loader, net, num_steps, population_code=False, num_classes=False):
      with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
    
        data_loader = iter(data_loader)
        for data, targets in data_loader:
          data = data.to(device)
          targets = targets.to(device)
          utils.reset(net)
          spk_rec, _ = net(data)
    
          if population_code:
            acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=10) * spk_rec.size(1)
          else:
            acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)
            
          total += spk_rec.size(1)
    
      return acc/total

Let’s run the training loop. Note that we are only training for
:math:`1` time step. I.e., each neuron only has the opportunity to fire
once. As a result, we might not expect the network to perform too well
here.

::

    from snntorch import backprop
    
    num_epochs = 5
    
    # training loop
    for epoch in range(num_epochs):
    
        avg_loss = backprop.BPTT(net, train_loader, num_steps=num_steps,
                              optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)
        
        print(f"Epoch: {epoch}")
        print(f"Test set accuracy: {test_accuracy(test_loader, net, num_steps)*100:.3f}%\n")

        >> Epoch: 0
        >> Test set accuracy: 59.421%

        >> Epoch: 1
        >> Test set accuracy: 61.889%

While there are ways to improve single time-step performance, e.g., by
applying the loss to the membrane potential, one single time-step is
extremely challenging to train a network competitively using rate codes.

With population coding
^^^^^^^^^^^^^^^^^^^^^^

Let’s modify the loss function to specify that population coding should
be enabled. We must also specify the number of classes. This means that
there will be a total of
:math:`50~neurons~per~class~=~500~neurons~/~10~classes`.

::

    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=10)
    optimizer = torch.optim.Adam(net_pop.parameters(), lr=2e-3, betas=(0.9, 0.999))

::

    num_epochs = 5
    
    # training loop
    for epoch in range(num_epochs):
    
        avg_loss = backprop.BPTT(net_pop, train_loader, num_steps=num_steps,
                                optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)
    
        print(f"Epoch: {epoch}")
        print(f"Test set accuracy: {test_accuracy(test_loader, net_pop, num_steps, population_code=True, num_classes=10)*100:.3f}%\n")

        >> Epoch: 0
        >> Test set accuracy: 80.501%

        >> Epoch: 1
        >> Test set accuracy: 82.690%

Even though we are only training on one time-step, introducing
additional output neurons has immediately enabled better performance.

Conclusion
^^^^^^^^^^

The performance boost from population coding may start to fade as the
number of time steps increases. But it may also be preferable to
increasing time steps as PyTorch is optimized for handling matrix-vector
products, rather than sequential, step-by-step operations over time.

-  For a detailed tutorial of spiking neurons, neural nets, encoding,
   and training using neuromorphic datasets, check out the `snnTorch
   tutorial
   series <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__.
-  For more information on the features of snnTorch, check out the
   `documentation at this
   link <https://snntorch.readthedocs.io/en/latest/>`__.
-  If you have ideas, suggestions or would like to find ways to get
   involved, then `check out the snnTorch GitHub project
   here. <https://github.com/jeshraghian/snntorch>`__


============================
Regression with SNNs: Part I
============================

Learning Membrane Potentials with LIF Neurons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tutorial written by Alexander Henkes (`ORCID <https://orcid.org/0000-0003-4615-9271>`_) and Jason K. Eshraghian (`ncg.ucsc.edu <https://ncg.ucsc.edu>`_)


.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb


This tutorial is based on the following papers on nonlinear regression
and spiking neural networks. If you find these resources or code useful
in your work, please consider citing the following sources:

   `Alexander Henkes, Jason K. Eshraghian, and Henning Wessels. “Spiking
   neural networks for nonlinear regression”, arXiv preprint
   arXiv:2210.03515, October 2022. <https://arxiv.org/abs/2210.03515>`_

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


In the regression tutorial series, you will learn how to use snnTorch to
perform regression using a variety of spiking neuron models, including:

-  Leaky Integrate-and-Fire (LIF) Neurons
-  Recurrent LIF Neurons
-  Spiking LSTMs

An overview of the regression tutorial series:

-  Part I (this tutorial) will train the membrane potential of a LIF
   neuron to follow a given trajectory over time.
-  Part II will use LIF neurons with recurrent feedback to perform classification using regression-based loss functions
-  Part III will use a more complex spiking LSTM network instead to train the firing time of a neuron.

::

    !pip install snntorch --quiet

::

    # imports
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import functional as SF
    from snntorch import utils
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import random
    import statistics
    import tqdm

Fix the random seed:

::

    # Seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

1. Spiking Regression
^^^^^^^^^^^^^^^^^^^^^

1.1 A Quick Background on Linear and Nonlinear Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tutorials so far have focused on multi-class classification
problems. But if you’ve made it this far, then it’s probably safe to say
that your brain can do more than distinguish cats and dogs. You’re
amazing and we believe in you.

An alternative problem is regression, where multiple input features
:math:`x_i` are used to estimate an output on a continuous number line
:math:`y \in \mathbb{R}`. A classic example is estimating the price of a
house, given a bunch of inputs such as land size, number of rooms, and
the local demand for avocado toast.

The objective of a regression problem is often the mean-square error:

.. math:: \mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2

or the mean absolute error:

.. math:: \mathcal{L}_{L1} = \frac{1}{n}\sum_{i=1}^n|y_i-\hat{y_i}|

where :math:`y` is the target and :math:`\hat{y}` is the predicted
value.

One of the challenges of linear regression is that it can only use
linear weightings of input features in predicting the output. Using a
neural network trained using the mean-square error as the cost function
allows us to perform nonlinear regression on more complex data.

1.2 Spiking Neurons in Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spikes are a type of nonlinearity that can also be used to learn more
complex regression tasks. But if spiking neurons only emit spikes that
are represented with 1’s and 0’s, then how might we perform regression?
I’m glad you asked! Here are a few ideas:

-  Use the total number of spikes (a rate-based code)
-  Use the time of the spike (a temporal/latency-based code)
-  Use the distance between pairs of spikes (i.e., using the interspike
   interval)

Or perhaps you pierce the neuron membrane with an electrical probe and
decide to use the membrane potential instead, which is a continuous
value.

   Note: is it cheating to directly access the membrane potential, i.e.,
   something that is meant to be a ‘hidden state’? At this time, there
   isn’t much consensus in the neuromorphic community. Despite being a
   high precision variable in many models (and thus computationally
   expensive), the membrane potential is commonly used in loss functions
   as it is a more ‘continuous’ variable compared to discrete time steps
   or spike counts. While it costs more in terms of power and latency to
   operate on higher-precision values, the impact might be minor if you
   have a small output layer, or if the output does not need to be
   scaled by weights. It really is a task-specific and hardware-specific
   question.

2. Setting up the Regression Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.1 Create Dataset
^^^^^^^^^^^^^^^^^^

Let’s construct a simple toy problem. The following class returns the
function we are hoping to learn. If ``mode = "linear"``, a straight line
with a random slope is generated. If ``mode = "sqrt"``, then the square
root of this straight line is taken instead.

Our goal: train a leaky integrate-and-fire neuron such that its membrane
potential follows the sample over time.

::

    class RegressionDataset(torch.utils.data.Dataset):
        """Simple regression dataset."""
    
        def __init__(self, timesteps, num_samples, mode):
            """Linear relation between input and output"""
            self.num_samples = num_samples # number of generated samples
            feature_lst = [] # store each generated sample in a list
    
            # generate linear functions one by one
            for idx in range(num_samples):
                end = float(torch.rand(1)) # random final point
                lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # generate linear function from 0 to end
                feature = lin_vec.view(timesteps, 1)
                feature_lst.append(feature) # add sample to list
    
            self.features = torch.stack(feature_lst, dim=1) # convert list to tensor
    
            # option to generate linear function or square-root function
            if mode == "linear":
                self.labels = self.features * 1
    
            elif mode == "sqrt":
                slope = float(torch.rand(1))
                self.labels = torch.sqrt(self.features * slope)
    
            else:
                raise NotImplementedError("'linear', 'sqrt'")
    
        def __len__(self):
            """Number of samples."""
            return self.num_samples
    
        def __getitem__(self, idx):
            """General implementation, but we only have one sample."""
            return self.features[:, idx, :], self.labels[:, idx, :]


To see what a random sample looks like, run the following code-block:

::

    num_steps = 50
    num_samples = 1
    mode = "sqrt" # 'linear' or 'sqrt'
    
    # generate a single data sample
    dataset = RegressionDataset(timesteps=num_steps, num_samples=num_samples, mode=mode)
    
    # plot
    sample = dataset.labels[:, 0, 0]
    plt.plot(sample)
    plt.title("Target function to teach network")
    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-1.png?raw=true
        :align: center
        :width: 450

2.2 Create DataLoader
^^^^^^^^^^^^^^^^^^^^^

The Dataset objects created above load data into memory, and the
DataLoader will serve it up in batches. DataLoaders in PyTorch are a
handy interface for passing data into a network. They return an iterator
divided up into mini-batches of size ``batch_size``.

::

    batch_size = 1 # only one sample to learn
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)

3. Construct Model
^^^^^^^^^^^^^^^^^^

Let us try a simple network using only leaky integrate-and-fire layers
without recurrence. Subsequent tutorials will show how to use more
complex neuron types with higher-order recurrence. These architectures
should work just fine, if there is no strong time dependency in the
data, i.e., the next time step has weak dependence on the previous one.

A few notes on the architecture below:

-  Setting ``learn_beta=True`` enables the decay rate ``beta`` to be a
   learnable parameter
-  Each neuron has a unique, and randomly initialized threshold and
   decay rate
-  The output layer has the reset mechanism disabled by setting
   ``reset_mechanism="none"`` as we will not use any output spikes

::

    class Net(torch.nn.Module):
        """Simple spiking neural network in snntorch."""
    
        def __init__(self, timesteps, hidden):
            super().__init__()
            
            self.timesteps = timesteps # number of time steps to simulate the network
            self.hidden = hidden # number of hidden neurons 
            spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
            
            # randomly initialize decay rate and threshold for layer 1
            beta_in = torch.rand(self.hidden)
            thr_in = torch.rand(self.hidden)
    
            # layer 1
            self.fc_in = torch.nn.Linear(in_features=1, out_features=self.hidden)
            self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)
            
            # randomly initialize decay rate and threshold for layer 2
            beta_hidden = torch.rand(self.hidden)
            thr_hidden = torch.rand(self.hidden)
    
            # layer 2
            self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
            self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)
    
            # randomly initialize decay rate for output neuron
            beta_out = torch.rand(1)
            
            # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
            self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=1)
            self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
    
        def forward(self, x):
            """Forward pass for several time steps."""
    
            # Initalize membrane potential
            mem_1 = self.lif_in.init_leaky()
            mem_2 = self.lif_hidden.init_leaky()
            mem_3 = self.li_out.init_leaky()
    
            # Empty lists to record outputs
            mem_3_rec = []
    
            # Loop over 
            for step in range(self.timesteps):
                x_timestep = x[step, :, :]
    
                cur_in = self.fc_in(x_timestep)
                spk_in, mem_1 = self.lif_in(cur_in, mem_1)
                
                cur_hidden = self.fc_hidden(spk_in)
                spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)
    
                cur_out = self.fc_out(spk_hidden)
                _, mem_3 = self.li_out(cur_out, mem_3)
    
                mem_3_rec.append(mem_3)
    
            return torch.stack(mem_3_rec)

Instantiate the network below:

::

    hidden = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = Net(timesteps=num_steps, hidden=hidden).to(device)


Let’s observe the behavior of the output neuron before it has been
trained and how it compares to the target function:

::

    train_batch = iter(dataloader)
    
    # run a single forward-pass
    with torch.no_grad():
        for feature, label in train_batch:
            feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
            label = torch.swapaxes(input=label, axis0=0, axis1=1)
            feature = feature.to(device)
            label = label.to(device)
            mem = model(feature)
    
    # plot
    plt.plot(mem[:, 0, 0].cpu(), label="Output")
    plt.plot(label[:, 0, 0].cpu(), '--', label="Target")
    plt.title("Untrained Output Neuron")
    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    plt.legend(loc='best')
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-2.png?raw=true
        :align: center
        :width: 450

As the network has not yet been trained, it is unsurprising the membrane
potential follows a senseless evolution.

4. Construct Training Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^

We call ``torch.nn.MSELoss()`` to minimize the mean square error between
the membrane potential and the target evolution.

We iterate over the same sample of data.

::

    num_iter = 100 # train for 100 iterations
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()
    
    loss_hist = [] # record loss
    
    # training loop
    with tqdm.trange(num_iter) as pbar:
        for _ in pbar:
            train_batch = iter(dataloader)
            minibatch_counter = 0
            loss_epoch = []
            
            for feature, label in train_batch:
                # prepare data
                feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
                label = torch.swapaxes(input=label, axis0=0, axis1=1)
                feature = feature.to(device)
                label = label.to(device)
    
                # forward pass
                mem = model(feature)
                loss_val = loss_function(mem, label) # calculate loss
                optimizer.zero_grad() # zero out gradients
                loss_val.backward() # calculate gradients
                optimizer.step() # update weights
    
                # store loss
                loss_hist.append(loss_val.item())
                loss_epoch.append(loss_val.item())
                minibatch_counter += 1
    
                avg_batch_loss = sum(loss_epoch) / minibatch_counter # calculate average loss p/epoch
                pbar.set_postfix(loss="%.3e" % avg_batch_loss) # print loss p/batch

5. Evaluation
^^^^^^^^^^^^^

::

    loss_function = torch.nn.L1Loss() # Use L1 loss instead
    
     # pause gradient calculation during evaluation
    with torch.no_grad():
        model.eval()
    
        test_batch = iter(dataloader)
        minibatch_counter = 0
        rel_err_lst = []
    
        # loop over data samples
        for feature, label in test_batch:
    
            # prepare data
            feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
            label = torch.swapaxes(input=label, axis0=0, axis1=1)
            feature = feature.to(device)
            label = label.to(device)
    
            # forward-pass
            mem = model(feature)
    
            # calculate relative error
            rel_err = torch.linalg.norm(
                (mem - label), dim=-1
            ) / torch.linalg.norm(label, dim=-1)
            rel_err = torch.mean(rel_err[1:, :])
    
            # calculate loss
            loss_val = loss_function(mem, label)
    
            # store loss
            loss_hist.append(loss_val.item())
            rel_err_lst.append(rel_err.item())
            minibatch_counter += 1
    
        mean_L1 = statistics.mean(loss_hist)
        mean_rel = statistics.mean(rel_err_lst)
    
    print(f"{'Mean L1-loss:':<{20}}{mean_L1:1.2e}")
    print(f"{'Mean rel. err.:':<{20}}{mean_rel:1.2e}")


::

    >> Mean L1-loss:       1.22e-02
    >> Mean rel. err.:     2.84e-02

Let’s plot our results for some visual intuition:

::

    mem = mem.cpu()
    label = label.cpu()
    
    plt.title("Trained Output Neuron")
    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    for i in range(batch_size):
        plt.plot(mem[:, i, :].cpu(), label="Output")
        plt.plot(label[:, i, :].cpu(), label="Target")
    plt.legend(loc='best')
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-3.png?raw=true
        :align: center
        :width: 450

It is a little jagged, but it’s not looking too bad.

You might try to improve the curve fit by expanding the size of the
hidden layer, increasing the number of iterations, adding extra time
steps, hyperparameter fine-tuning, or using a completely different
neuron type.

Conclusion
^^^^^^^^^^

The next regression tutorials will test more powerful spiking neurons,
such as Recurrent LIF neurons and spiking LSTMs, to see how they
compare.

If you like this project, please consider starring ⭐ the repo on GitHub
as it is the easiest and best way to support it.

Additional Resources
^^^^^^^^^^^^^^^^^^^^

-  `Check out the snnTorch GitHub project
   here. <https://github.com/jeshraghian/snntorch>`__
-  More detail on nonlinear regression with SNNs can be found in our
   corresponding preprint here: `Henkes, A.; Eshraghian, J. K.; and
   Wessels, H. “Spiking neural networks for nonlinear regression”, arXiv
   preprint arXiv:2210.03515,
   Oct. 2022. <https://arxiv.org/abs/2210.03515>`__


=============================
Regression with SNNs: Part II
=============================

Regression-based Classification with Recurrent Leaky Integrate-and-Fire Neurons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tutorial written by Alexander Henkes (`ORCID <https://orcid.org/0000-0003-4615-9271>`_) and Jason K. Eshraghian (`ncg.ucsc.edu <https://ncg.ucsc.edu>`_)


.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb


This tutorial is based on the following papers on nonlinear regression
and spiking neural networks. If you find these resources or code useful
in your work, please consider citing the following sources:

   `Alexander Henkes, Jason K. Eshraghian, and Henning Wessels. “Spiking
   neural networks for nonlinear regression”, arXiv preprint
   arXiv:2210.03515, October 2022. <https://arxiv.org/abs/2210.03515>`_

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


In the regression tutorial series, you will learn how to use snnTorch to
perform regression using a variety of spiking neuron models, including:

-  Leaky Integrate-and-Fire (LIF) Neurons
-  Recurrent LIF Neurons
-  Spiking LSTMs

An overview of the regression tutorial series:

-  Part I will train the membrane potential of a LIF neuron to follow a
   given trajectory over time.
-  Part II (this tutorial) will use LIF neurons with recurrent feedback
   to perform classification using regression-based loss functions
-  Part III will use a more complex spiking LSTM network instead to
   train the firing time of a neuron.


::

    !pip install snntorch --quiet

::

    # imports
    import snntorch as snn
    from snntorch import functional as SF
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import tqdm

1. Classification as Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In conventional deep learning, we often calculate the Cross Entropy Loss
to train a network to do classification. The output neuron with the
highest activation is thought of as the predicted class.

In spiking neural nets, this may be interpreted as the class that fires
the most spikes. I.e., apply cross entropy to the total spike count (or
firing frequency). The effect of this is that the predicted class will
be maximized, while other classes aim to be suppressed.

The brain does not quite work like this. SNNs are sparsely activated,
and while approaching SNNs with this deep learning attitude may lead to
optimal accuracy, it’s important not to ‘overfit’ too much to what the
deep learning folk are doing. After all, we use spikes to achieve better
power efficiency. Good power efficiency relies on sparse spiking
activity.

In other words, training bio-inspired SNNs using deep learning tricks
does not lead to brain-like activity.

So what can we do?

We will focus on recasting classification problems into regression
tasks. This is done by training the predicted neuron to fire a given
number of times, while incorrect neurons are trained to still fire a
given number of times, albeit less frequently.

This contrasts with cross-entropy which would try to drive the correct
class to fire at *all* time steps, and incorrect classes to not fire at
all.

As with the previous tutorial, we can use the mean-square error to
achieve this. Recall the form of the mean-square error loss:

.. math:: \mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2

where :math:`y` is the target and :math:`\hat{y}` is the predicted
value.

To apply MSE to the spike count, assume we have :math:`n` output neurons
in a classification problem, where :math:`n` is the number of possible
classes. :math:`\hat{y}_i` is now the total number of spikes the
:math:`i^{th}` output neuron emits over the full simulation runtime.

Given that we have :math:`n` neurons, this means that :math:`y` and
:math:`\hat{y}` must be vectors with :math:`n` elements, and our loss
will sum the independent MSE losses of each neuron.

1.1 A Theoretical Example
^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a simulation of 10 time steps. Say we wish for the correct
neuron class to fire 8 times, and the incorrect classes to fire 2 times.
Assume :math:`y_1` is the correct class:

.. math::  y = \begin{bmatrix} 8 \\ 2 \\ \vdots \\ 2 \end{bmatrix},  \hat{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}

The element-wise MSE is taken to generate :math:`n` loss components,
which are all summed together to generate a final loss.

2. Recurrent Leaky Integrate-and-Fire Neurons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neurons in the brain have a ton of feedback connections. And so the SNN
community have been exploring the dynamics of networks that feed output
spikes back to the input. This is in addition to the recurrent dynamics
of the membrane potential.

There are a few ways to construct recurrent leaky integrate-and-fire
(``RLeaky``) neurons in snnTorch. Refer to the
`docs <https://snntorch.readthedocs.io/en/latest/snn.neurons_rleaky.html>`__
for an exhaustive description of the neuron’s hyperparameters. Let’s see
a few examples.

2.1 RLIF Neurons with 1-to-1 connections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression2/reg2-1.jpg?raw=true
        :align: center
        :width: 400

This assumes each neuron feeds back its output spikes into itself, and
only itself. There are no cross-coupled connections between neurons in
the same layer.

::

    beta = 0.9 # membrane potential decay rate
    num_steps = 10 # 10 time steps
    
    rlif = snn.RLeaky(beta=beta, all_to_all=False) # initialize RLeaky Neuron
    spk, mem = rlif.init_rleaky() # initialize state variables
    x = torch.rand(1) # generate random input
    
    spk_recording = []
    mem_recording = []
    
    # run simulation
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

By default, ``V`` is a learnable parameter that initializes to :math:`1`
and will be updated during the training process. If you wish to disable
learning, or use your own initialization variables, then you may do so
as follows:

::

    rlif = snn.RLeaky(beta=beta, all_to_all=False, learn_recurrent=False) # disable learning of recurrent connection
    rlif.V = torch.rand(1) # set this to layer size
    print(f"The recurrent weight is: {rlif.V.item()}")

2.2 RLIF Neurons with all-to-all connections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.2.1 Linear feedback
^^^^^^^^^^^^^^^^^^^^^


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression2/reg2-2.jpg?raw=true
        :align: center
        :width: 400

By default, ``RLeaky`` assumes feedback connections where all spikes
from a given layer are first weighted by a feedback layer before being
passed to the input of all neurons. This introduces more parameters, but
it is thought this helps with learning time-varying features in data.

::

    beta = 0.9 # membrane potential decay rate
    num_steps = 10 # 10 time steps
    
    rlif = snn.RLeaky(beta=beta, linear_features=10)  # initialize RLeaky Neuron
    spk, mem = rlif.init_rleaky() # initialize state variables
    x = torch.rand(10) # generate random input
    
    spk_recording = []
    mem_recording = []
    
    # run simulation
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

You can disable learning in the feedback layer with
``learn_recurrent=False``.

2.2.2 Convolutional feedback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using a convolutional layer, this will throw an error because
it does not make sense for the output spikes (3-dimensional) to be
projected into 1-dimension by a ``nn.Linear`` feedback layer.

To address this, you must specify that you are using a convolutional
feedback layer:

::

    beta = 0.9 # membrane potential decay rate
    num_steps = 10 # 10 time steps
    
    rlif = snn.RLeaky(beta=beta, conv2d_channels=3, kernel_size=(5,5))  # initialize RLeaky Neuron
    spk, mem = rlif.init_rleaky() # initialize state variables
    x = torch.rand(3, 32, 32) # generate random 3D input
    
    spk_recording = []
    mem_recording = []
    
    # run simulation
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

To ensure the output spike dimension matches the input dimensions,
padding is automatically applied.

If you have exotically shaped data, you will need to construct your own
feedback layers manually.

3. Construct Model
^^^^^^^^^^^^^^^^^^

Let’s train a couple of models using ``RLeaky`` layers. For speed, we
will train a model with linear feedback.

::

    class Net(torch.nn.Module):
        """Simple spiking neural network in snntorch."""
    
        def __init__(self, timesteps, hidden, beta):
            super().__init__()
            
            self.timesteps = timesteps
            self.hidden = hidden
            self.beta = beta
    
            # layer 1
            self.fc1 = torch.nn.Linear(in_features=784, out_features=self.hidden)
            self.rlif1 = snn.RLeaky(beta=self.beta, linear_features=self.hidden)
    
            # layer 2
            self.fc2 = torch.nn.Linear(in_features=self.hidden, out_features=10)
            self.rlif2 = snn.RLeaky(beta=self.beta, linear_features=10)
    
        def forward(self, x):
            """Forward pass for several time steps."""
    
            # Initalize membrane potential
            spk1, mem1 = self.rlif1.init_rleaky()
            spk2, mem2 = self.rlif2.init_rleaky()
    
            # Empty lists to record outputs
            spk_recording = []
    
            for step in range(self.timesteps):
                spk1, mem1 = self.rlif1(self.fc1(x), spk1, mem1)
                spk2, mem2 = self.rlif2(self.fc2(spk1), spk2, mem2)
                spk_recording.append(spk2)
    
            return torch.stack(spk_recording)

Instantiate the network below:

::

    hidden = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = Net(timesteps=num_steps, hidden=hidden, beta=0.9).to(device)

4. Construct Training Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^

4.1 Mean Square Error Loss in ``snntorch.functional``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From ``snntorch.functional``, we call ``mse_count_loss`` to set the
target neuron to fire 80% of the time, and incorrect neurons to fire 20%
of the time. What it took 10 paragraphs to explain is achieved in one
line of code:

::

    loss_function = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

4.2 DataLoader
^^^^^^^^^^^^^^

Dataloader boilerplate. Let’s just do MNIST, and testing this on
temporal data is an exercise left to the reader/coder.

::

    batch_size = 128
    data_path='/tmp/data/mnist'
    
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

4.3 Train Network
^^^^^^^^^^^^^^^^^

::

    num_epochs = 5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_hist = []
    
    with tqdm.trange(num_epochs) as pbar:
        for _ in pbar:
            train_batch = iter(train_loader)
            minibatch_counter = 0
            loss_epoch = []
    
            for feature, label in train_batch:
                feature = feature.to(device)
                label = label.to(device)
    
                spk = model(feature.flatten(1)) # forward-pass
                loss_val = loss_function(spk, label) # apply loss
                optimizer.zero_grad() # zero out gradients
                loss_val.backward() # calculate gradients
                optimizer.step() # update weights
    
                loss_hist.append(loss_val.item())
                minibatch_counter += 1
    
                avg_batch_loss = sum(loss_hist) / minibatch_counter
                pbar.set_postfix(loss="%.3e" % avg_batch_loss)

5. Evaluation
^^^^^^^^^^^^^

::

    test_batch = iter(test_loader)
    minibatch_counter = 0
    loss_epoch = []
    
    model.eval()
    with torch.no_grad():
      total = 0
      acc = 0
      for feature, label in test_batch:
          feature = feature.to(device)
          label = label.to(device)
    
          spk = model(feature.flatten(1)) # forward-pass
          acc += SF.accuracy_rate(spk, label) * spk.size(1)
          total += spk.size(1)
    
    print(f"The total accuracy on the test set is: {(acc/total) * 100:.2f}%")

6. Alternative Loss Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous tutorial, we tested membrane potential learning. We can
do the same here by setting the target neuron to reach a membrane
potential greater than the firing threshold, and incorrect neurons to
reach a membrane potential below the firing threshold:

::

    loss_function = SF.mse_membrane_loss(on_target=1.05, off_target=0.2)

In the above case, we are trying to get the correct neuron to constantly
sit above the firing threshold.

Try updating the network and the training loop to make this work.

Hints: 

- You will need to return the output membrane potential instead of spikes. 

- Pass membrane potential to the loss function instead of spikes

Conclusion
^^^^^^^^^^

The next regression tutorial will introduce spiking LSTMs to achieve
precise spike time learning.

If you like this project, please consider starring ⭐ the repo on GitHub
as it is the easiest and best way to support it.

Additional Resources
^^^^^^^^^^^^^^^^^^^^

-  `Check out the snnTorch GitHub project
   here. <https://github.com/jeshraghian/snntorch>`__
-  More detail on nonlinear regression with SNNs can be found in our
   corresponding preprint here: `Henkes, A.; Eshraghian, J. K.; and
   Wessels, H. “Spiking neural networks for nonlinear regression”, arXiv
   preprint arXiv:2210.03515,
   Oct. 2022. <https://arxiv.org/abs/2210.03515>`__


.. container:: cell markdown

   ` <https://github.com/jeshraghian/snntorch/>`__

   .. rubric:: snnTorch - Spiking Autoencoder (SAE) using Convolutional
      Spiking Neural Networks
      :name: snntorch---spiking-autoencoder-sae-using-convolutional-spiking-neural-networks

   .. rubric:: Tutorial by Alon Loeffler (www.alonloeffler.com)
      :name: tutorial-by-alon-loeffler-wwwalonloefflercom

   \*This tutorial is adapted from my original article published on
   Medium.com

   ` <https://github.com/jeshraghian/snntorch/>`__
   ` <https://github.com/jeshraghian/snntorch/>`__

.. container:: cell markdown

   For a comprehensive overview on how SNNs work, and what is going on
   under the hood, `then you might be interested in the snnTorch
   tutorial series available
   here. <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__
   The snnTorch tutorial series is based on the following paper. If you
   find these resources or code useful in your work, please consider
   citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_
.. container:: cell markdown

   In this tutorial, you will learn how to use snnTorch to:

   -  Create a spiking Autoencoder
   -  Reconstruct MNIST images

   If running in Google Colab:

   -  You may connect to GPU by checking ``Runtime`` >
      ``Change runtime type`` > ``Hardware accelerator: GPU``

.. container:: cell markdown

   .. rubric:: 1. Autoencoders
      :name: 1-autoencoders

   | An autoencoder is a neural network that is trained to reconstruct
     its input data. It consists of two main components: 1) An encoder
   | 2) A decoder

   The encoder takes in input data (e.g. an image) and maps it to a
   lower-dimensional latent space. For example an encoder might take in
   as input a 28 x 28 pixel MNIST image (784 pixels total), and extract
   the important features from the image while compressing it to a
   smaller dimensionality (e.g. 32 features). This compressed
   representation of the image is called the *latent representation*.

   The decoder maps the latent representation back to the original input
   space (i.e. from 32 features back to 784 pixels), and tries to
   reconstruct the original image from a small number of key features.

   .. raw:: html

      <center>
         <figure>
      <img src='https://miro.medium.com/max/828/0*dHxZ5LCq5kEPltWH.webp' width="600">
      <figcaption> Example of a simple Autoencoder where x is the input data, z is the encoded latent space, and x' is the reconstructed inputs once z is decoded (source: Wikipedia). </figcaption>
                  </figure>
      </center>

   The goal of the autoencoder is to minimize the reconstruction error
   between the input data and the output of the decoder.

   This is achieved by training the model to minimize the reconstruction
   loss, which is typically defined as the mean squared error (MSE)
   between the input and the reconstructed output.

   .. raw:: html

      <center>
         <figure>
      <img src='https://miro.medium.com/max/640/1*kjfms6RCnHVMLRSq75AD0Q.webp'>
      <figcaption> MSE loss equation. Here, $y$ would represent the original image (y true) and $\hat{y}$ would represent the reconstructed outputs (y pred) (source: Towards Data Science). </figcaption>
                  </figure>
      </center>

   Autoencoders are excellent tools for reducing noise in data by
   finding only the important parts of the data, and discarding
   everything else during the reconstruction process. This is
   effectively a dimensionality reduction tool.

.. container:: cell markdown

   In this tutorial (similar to tutorial 1), we will assume we have some
   non-spiking input data (i.e., the MNIST dataset) and that we want to
   encode it and reconstruct it. So let's get started!

.. container:: cell markdown

   .. rubric:: 2. Setting Up
      :name: 2-setting-up

.. container:: cell markdown

   .. rubric:: 2.1 Install/Import packages and set up environment
      :name: 21-installimport-packages-and-set-up-environment

.. container:: cell markdown

   To start, we need to install snnTorch and its dependencies (note this
   tutorial assumes you have pytorch and torchvision already installed -
   these come preinstalled in Colab). You can do this by running the
   following command:

.. container:: cell code

   .. code:: python

      !pip install snntorch

.. container:: cell markdown

   Next, let’s import the necessary modules and set up the SAE model.

   We can use pyTorch to define the encoder and decoder networks, and
   snnTorch to convert the neurons in the networks into leaky integrate
   and fire (LIF) neurons, which read in and output spikes.

   We will be using convolutional neural networks (CNN), covered in
   tutorial 6, for the basis of our encoder and decoder.

.. container:: cell code

   .. code:: python

      import os

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      from torchvision import datasets, transforms
      from torch.utils.data import DataLoader
      from torchvision import utils as utls

      import snntorch as snn
      from snntorch import utils
      from snntorch import surrogate

      import numpy as np

      #Define the SAE model:
      class SAE(nn.Module):
          def __init__(self,latent_dim):
              super().__init__()
              self.latent_dim = latent_dim #dimensions of the encoded z-space data

.. container:: cell markdown

   .. rubric:: 3. Building the Autoencoder
      :name: 3-building-the-autoencoder

.. container:: cell markdown

   .. rubric:: 3.1 DataLoaders
      :name: 31-dataloaders

   We will be using the MNIST dataset

.. container:: cell code

   .. code:: python

      # dataloader arguments
      batch_size = 250
      data_path='/tmp/data/mnist'

      dtype = torch.float
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

.. container:: cell code

   .. code:: python

      # Define a transform
      input_size = 32 #for the sake of this tutorial, we will be resizing the original MNIST from 28 to 32

      transform = transforms.Compose([
                  transforms.Resize((input_size, input_size)),
                  transforms.Grayscale(),
                  transforms.ToTensor(),
                  transforms.Normalize((0,), (1,))])

      # Load MNIST

      # Training data
      train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

      # Testing data
      test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transform, download=True)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

.. container:: cell markdown

   .. rubric:: 3.2 The Encoder
      :name: 32-the-encoder

   Let's start building the sections of our autoencoder which we slowly
   combine together to the SAE model we defined above:

.. container:: cell markdown

   First, let's add an encoder with three convolutional layers
   (``nn.Conv2d``), and one fully-connected linear output layer.

   -  We will use a kernel of size 3, with padding of 1 and stride of 2
      for the CNN hyperparameters.

   -  We also add a Batch Norm layer between convolutional layers. Since
      will be using the neuron membrane potential as outputs from each
      neuron, normalization will help our training process.

.. container:: cell code

   .. code:: python

      #Define the SAE model:
      class SAE(nn.Module):
          def __init__(self):
              super().__init__()
              self.latent_dim = latent_dim #dimensions of the encoded z-space data
              
              # Encoder
              self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3,padding = 1,stride=2), # Conv Layer 1
                                  nn.BatchNorm2d(32),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh), #SNN TORCH LIF NEURON
                                  nn.Conv2d(32, 64, 3,padding = 1,stride=2), # Conv Layer 2
                                  nn.BatchNorm2d(64),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.Conv2d(64, 128, 3,padding = 1,stride=2), # Conv Layer 3
                                  nn.BatchNorm2d(128),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.Flatten(start_dim = 1, end_dim = 3), #Flatten convolutional output
                                  nn.Linear(128*4*4, latent_dim), # Fully connected linear layer
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh)
                                  )

.. container:: cell markdown

   .. rubric:: 3.3 The Decoder
      :name: 33-the-decoder

.. container:: cell markdown

   Before we write the decoder, there is one more small step required.
   When decoding the latent z-space data, we need to move from the
   flattened encoded representation (latent_dim) back to a tensor
   representation to use in transposed convolution.

   To do so, we need to run an additional fully-connected linear layer
   transforming the data back into a tensor of 128 x 4 x 4.

.. container:: cell code

   .. code:: python

      #Define the SAE model:
      class SAE(nn.Module):
          def __init__(self,latent_dim):
              super().__init__()
              self.latent_dim = latent_dim #dimensions of the encoded z-space data
              
              # Encoder
              self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3,padding = 1,stride=2), # Conv Layer 1
                                  nn.BatchNorm2d(32),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh), #SNN TORCH LIF NEURON
                                  nn.Conv2d(32, 64, 3,padding = 1,stride=2), # Conv Layer 2
                                  nn.BatchNorm2d(64),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.Conv2d(64, 128, 3,padding = 1,stride=2), # Conv Layer 3
                                  nn.BatchNorm2d(128),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.Flatten(start_dim = 1, end_dim = 3), #Flatten convolutional output
                                  nn.Linear(128*4*4, latent_dim), # Fully connected linear layer
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh)
                                  )

              # From latent back to tensor for convolution
              self.linearNet= nn.Sequential(nn.Linear(latent_dim,128*4*4),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh))

.. container:: cell markdown

   Now we can write the decoder, with three transposed convolutional
   (``nn.ConvTranspose2d``) layers and one linear output layer. Although
   we converted the latent data back into tensor form for convolution,
   we still need to Unflatten it to a tensor of 128 x 4 x 4, as the
   input to the network is 1 dimensional. This is done using
   ``nn.Unflatten`` in the first line of the Decoder.

.. container:: cell code

   .. code:: python

      #Define the SAE model:
      class SAE(nn.Module):
          def __init__(self,latent_dim):
              super().__init__()
              self.latent_dim = latent_dim #dimensions of the encoded z-space data
              
              # Encoder
              self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3,padding = 1,stride=2), # Conv Layer 1
                                  nn.BatchNorm2d(32),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh), #SNN TORCH LIF NEURON
                                  nn.Conv2d(32, 64, 3,padding = 1,stride=2), # Conv Layer 2
                                  nn.BatchNorm2d(64),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.Conv2d(64, 128, 3,padding = 1,stride=2), # Conv Layer 3
                                  nn.BatchNorm2d(128),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.Flatten(start_dim = 1, end_dim = 3), #Flatten convolutional output
                                  nn.Linear(128*4*4, latent_dim), # Fully connected linear layer
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh)
                                  )

              # From latent back to tensor for convolution
              self.linearNet = nn.Sequential(nn.Linear(latent_dim,128*4*4),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh))
              # Decoder
              self.decoder = nn.Sequential(nn.Unflatten(1,(128,4,4)), #Unflatten data from 1 dim to tensor of 128 x 4 x 4
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.ConvTranspose2d(128, 64, 3,padding = 1,stride=(2,2),output_padding=1),
                                  nn.BatchNorm2d(64),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.ConvTranspose2d(64, 32, 3,padding = 1,stride=(2,2),output_padding=1),
                                  nn.BatchNorm2d(32),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                  nn.ConvTranspose2d(32, 1, 3,padding = 1,stride=(2,2),output_padding=1),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,output=True,threshold=20000) #make large so membrane can be trained
                                  )

.. container:: cell markdown

   One important thing to note is in the final Leaky layer, our spiking
   threshold (``thresh``) is set extremely high. This is a neat trick in
   snnTorch, which allows the neuron membrane in the final layer to
   continuously be updated, without ever reaching a spiking threshold.

   The output of each Leaky Neuron will consist of a tensor of spikes (0
   or 1) and a tensor of neuron membrane potential (negative or positive
   real numbers). snnTorch allows us to use either the spikes or
   membrane potential of each neuron in training. We will be using the
   membrane potential output from the final layer for the image
   reconstruction.

.. container:: cell markdown

   .. rubric:: 3.4 Forward Function
      :name: 34-forward-function

   Finally, let’s write the forward, encode and decode functions, before
   putting it all together

.. container:: cell code

   .. code:: python

      def forward(self, x): 
          utils.reset(self.encoder) #need to reset the hidden states of LIF 
          utils.reset(self.decoder)
          utils.reset(self.linearNet) 
          
          #encode
          spk_mem=[];spk_rec=[];encoded_x=[]
          for step in range(num_steps): #for t in time
              spk_x,mem_x=self.encode(x) #Output spike trains and neuron membrane states
              spk_rec.append(spk_x) 
              spk_mem.append(mem_x)
          spk_rec=torch.stack(spk_rec,dim=2) # stack spikes in second tensor dimension
          spk_mem=torch.stack(spk_mem,dim=2) # stack membranes in second tensor dimension
          
          #decode
          spk_mem2=[];spk_rec2=[];decoded_x=[]
          for step in range(num_steps): #for t in time
              x_recon,x_mem_recon=self.decode(spk_rec[...,step]) 
              spk_rec2.append(x_recon) 
              spk_mem2.append(x_mem_recon)
          spk_rec2=torch.stack(spk_rec2,dim=4)
          spk_mem2=torch.stack(spk_mem2,dim=4)  
          out = spk_mem2[:,:,:,:,-1] #return the membrane potential of the output neuron at t = -1 (last t)
          return out 

      def encode(self,x):
          spk_latent_x,mem_latent_x=self.encoder(x) 
          return spk_latent_x,mem_latent_x

      def decode(self,x):
          spk_x,mem_x = self.latentToConv(x) #convert latent dimension back to total size of features in encoder final layer
          spk_x2,mem_x2=self.decoder(spk_x)
          return spk_x2,mem_x2   

.. container:: cell markdown

   There are a couple of key things to notice here:

   1) At the beginning of each call of our forward function, we need to
   reset the hidden weights of each LIF neuron. If we do not do this, we
   will get weird gradient errors from pytorch when we try to backprop.
   To do so we use ``utils.reset``.

   2) In the forward function, when we call the encode and decode
   functions, we do so in a loop. This is because we are converting
   static images into spike trains, as explained previously. Spike
   trains need a time, t, during which spiking can occur or not occur.
   Therefore, we encode and decode the original image :math:`t` (or
   ``num_steps``) times, to create a latent representation, :math:`z`.

.. container:: cell markdown

   For example, converting a sample digit 7 from the MNIST dataset into
   a spike-train with a latent dimension of 32 and t = 50, might look
   like this: Spike-Train of sample MNIST digit 7 after encoding. Other
   instances of 7 will have slightly different spike-trains, and
   different digits will have even more different spike-trains.

.. container:: cell markdown

   .. rubric:: 3.5 Putting it all together:
      :name: 35-putting-it-all-together

   Our final, complete SAE class should look like this:

.. container:: cell code

   .. code:: python

      class SAE(nn.Module):
          def __init__(self):
              super().__init__()
              #Encoder
              self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3,padding = 1,stride=2),
                                nn.BatchNorm2d(32),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                nn.Conv2d(32, 64, 3,padding = 1,stride=2),
                                nn.BatchNorm2d(64),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                nn.Conv2d(64, 128, 3,padding = 1,stride=2),
                                nn.BatchNorm2d(128),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                nn.Flatten(start_dim = 1, end_dim = 3),
                                nn.Linear(2048, latent_dim), #this needs to be the final layer output size (channels * pixels * pixels)
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh)
                                )
             # From latent back to tensor for convolution
              self.linearNet= nn.Sequential(nn.Linear(latent_dim,128*4*4),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh))        #Decoder
              
              self.decoder = nn.Sequential(nn.Unflatten(1,(128,4,4)), 
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                nn.ConvTranspose2d(128, 64, 3,padding = 1,stride=(2,2),output_padding=1),
                                nn.BatchNorm2d(64),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                nn.ConvTranspose2d(64, 32, 3,padding = 1,stride=(2,2),output_padding=1),
                                nn.BatchNorm2d(32),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                                nn.ConvTranspose2d(32, 1, 3,padding = 1,stride=(2,2),output_padding=1),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,output=True,threshold=20000) #make large so membrane can be trained
                                )
              
          def forward(self, x): #Dimensions: [Batch,Channels,Width,Length]
              utils.reset(self.encoder) #need to reset the hidden states of LIF 
              utils.reset(self.decoder)
              utils.reset(self.linearNet) 
              
              #encode
              spk_mem=[];spk_rec=[];encoded_x=[]
              for step in range(num_steps): #for t in time
                  spk_x,mem_x=self.encode(x) #Output spike trains and neuron membrane states
                  spk_rec.append(spk_x) 
                  spk_mem.append(mem_x)
              spk_rec=torch.stack(spk_rec,dim=2)
              spk_mem=torch.stack(spk_mem,dim=2) #Dimensions:[Batch,Channels,Width,Length, Time]
              
              #decode
              spk_mem2=[];spk_rec2=[];decoded_x=[]
              for step in range(num_steps): #for t in time
                  x_recon,x_mem_recon=self.decode(spk_rec[...,step]) 
                  spk_rec2.append(x_recon) 
                  spk_mem2.append(x_mem_recon)
              spk_rec2=torch.stack(spk_rec2,dim=4)
              spk_mem2=torch.stack(spk_mem2,dim=4)#Dimensions:[Batch,Channels,Width,Length, Time]  
              out = spk_mem2[:,:,:,:,-1] #return the membrane potential of the output neuron at t = -1 (last t)
              return out #Dimensions:[Batch,Channels,Width,Length]

          def encode(self,x):
              spk_latent_x,mem_latent_x=self.encoder(x) 
              return spk_latent_x,mem_latent_x

          def decode(self,x):
              spk_x,mem_x = self.linearNet(x) #convert latent dimension back to total size of features in encoder final layer
              spk_x2,mem_x2=self.decoder(spk_x)
              return spk_x2,mem_x2

.. container:: cell markdown

   .. rubric:: 4. Training and Testing
      :name: 4-training-and-testing

   Finally, we can move on to training our SAE, and testing its
   usefulness. We have already loaded the MNIST dataset, and split it
   into training and testing classes.

.. container:: cell markdown

   .. rubric:: 4.1 Training Function
      :name: 41-training-function

   We define our training function, which takes in the network model,
   training dataset, optimizer and epoch number as inputs, and returns
   the loss value after running all batches of the current epoch.

   As discussed at the beginning, we will be using MSE loss to compare
   the reconstructed image (``x_recon``) with the original image
   (``real_img``)

   As always, to set up our gradients for backprop we use
   ``opti.zero_grad()``, and then call ``loss_val.backward()`` and
   ``opti.step()`` to perform backprop.

.. container:: cell code

   .. code:: python

      #Training 
      def train(network, trainloader, opti, epoch): 
          
          network=network.train()
          train_loss_hist=[]
          for batch_idx, (real_img, labels) in enumerate(trainloader):   
              opti.zero_grad()
              real_img = real_img.to(device)
              labels = labels.to(device)
              
              #Pass data into network, and return reconstructed image from Membrane Potential at t = -1
              x_recon = network(real_img) #Dimensions passed in: [Batch_size,Input_size,Image_Width,Image_Length] 
              
              #Calculate loss        
              loss_val = F.mse_loss(x_recon, real_img)
                      
              print(f'Train[{epoch}/{max_epoch}][{batch_idx}/{len(trainloader)}] Loss: {loss_val.item()}')

              loss_val.backward()
              opti.step()

              #Save reconstructed images every at the end of the epoch
              if batch_idx == len(trainloader)-1:
                  # NOTE: you need to create training/ and testing/ folders in your chosen path
                  utls.save_image((real_img+1)/2, f'figures/training/epoch{epoch}_finalbatch_inputs.png') 
                  utls.save_image((x_recon+1)/2, f'figures/training/epoch{epoch}_finalbatch_recon.png')
          return loss_val

.. container:: cell markdown

   .. rubric:: 4.2 Testing Function
      :name: 42-testing-function

   The testing function is nearly identifcal to the training function,
   except we do not backpropagate, therefore no gradients are required
   and we use ``torch.no_grad()``

.. container:: cell code

   .. code:: python

      #Testing 
      def test(network, testloader, opti, epoch):
          network=network.eval()
          test_loss_hist=[]
          with torch.no_grad(): #no gradient this time
              for batch_idx, (real_img, labels) in enumerate(testloader):   
                  real_img = real_img.to(device)#
                  labels = labels.to(device)
                  x_recon = network(real_img)

                  loss_val = F.mse_loss(x_recon, real_img)

                  print(f'Test[{epoch}/{max_epoch}][{batch_idx}/{len(testloader)}]  Loss: {loss_val.item()}')#, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')
                      
                  if batch_idx == len(testloader)-1:
                      utls.save_image((real_img+1)/2, f'figures/testing/epoch{epoch}_finalbatch_inputs.png')
                      utls.save_image((x_recon+1)/2, f'figures/testing/epoch{epoch}_finalbatch_recons.png')
          return loss_val

.. container:: cell markdown

   There are a couple of ways to calculate loss with spiking neural
   networks. Here, we are simply taking the membrane potential of the
   final fully-connected layer of neurons at the last time step
   (:math:`t = 5`).

   Therefore, we only need to compare each original image with its
   corresponding decoded, reconstructed image once per epoch. We can
   also return the membrane potentials at each time step, and create t
   different versions of the reconstructed image, and then compare each
   of them with the original image and take the average loss. For those
   of you interested in this, you can replace the loss function above
   with something like this:

   (*note this will fail to run as we have not defined any of the
   variables yet, it is just here for illustrative purposes*)

.. container:: cell code

   .. code:: python

      train_loss_hist=[]
      loss_val = torch.zeros((1), dtype=dtype, device=device)
      for step in range(num_steps):
          loss_val += F.mse_loss(x_recon, real_img)
      train_loss_hist.append(loss_val.item())
      avg_loss=loss_val/num_steps

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         NameError                                 Traceback (most recent call last)
         Cell In[72], line 4
               2 loss_val = torch.zeros((1), dtype=dtype, device=device)
               3 for step in range(num_steps):
         ----> 4     loss_val += F.mse_loss(x_recon, real_img)
               5 train_loss_hist.append(loss_val.item())
               6 avg_loss=loss_val/num_steps

         NameError: name 'x_recon' is not defined

.. container:: cell markdown

   .. rubric:: 5. Conclusion: Running the SAE
      :name: 5-conclusion-running-the-sae

   Now, finally, we can run our SAE model. Let’s define some parameters,
   and run training and testing

.. container:: cell markdown

   Let's create directories where we can save our original and
   reconstructed images for training and testing:

.. container:: cell code

   .. code:: python

      # create training/ and testing/ folders in your chosen path
      if not os.path.isdir('figures/training'):
          os.makedirs('figures/training')
          
      if not os.path.isdir('figures/testing'):
          os.makedirs('figures/testing')

.. container:: cell code

   .. code:: python

      # dataloader arguments
      batch_size = 250
      input_size = 32 #resize of mnist data (optional)

      #setup GPU
      dtype = torch.float
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

      # neuron and simulation parameters
      spike_grad = surrogate.atan(alpha=2.0)# alternate surrogate gradient fast_sigmoid(slope=25) 
      beta = 0.5 #decay rate of neurons 
      num_steps=5
      latent_dim = 32 #dimension of latent layer (how compressed we want the information)
      thresh=1#spiking threshold (lower = more spikes are let through)
      epochs=10 
      max_epoch=epochs

      #Define Network and optimizer
      net=SAE()
      net = net.to(device)

      optimizer = torch.optim.AdamW(net.parameters(), 
                                  lr=0.0001,
                                  betas=(0.9, 0.999), 
                                  weight_decay=0.001)

      #Run training and testing        
      for e in range(epochs): 
          train_loss = train(net, train_loader, optimizer, e)
          test_loss = test(net,test_loader,optimizer,e)

   .. container:: output stream stdout

      ::

         Train[0/10][0/240] Loss: 0.10109379142522812
         Train[0/10][1/240] Loss: 0.10465191304683685

   .. container:: output stream stderr

      ::


         KeyboardInterrupt

.. container:: cell markdown

   After only 10 epochs, our training and testing reconstructed losses
   should be around 0.05, and our reconstructed images should look
   something like this:

.. container:: cell markdown

.. container:: cell markdown

   Yes, the reconstructed images are a bit blurry, and the loss isn’t
   perfect, but from only 10 epochs, and only using the final membrane
   potential at :math:`t = 5` for our reconstructed loss, it’s a pretty
   decent start!

.. container:: cell markdown

   Try increasing the number of epochs, or playing around with
   ``thresh``, ``num_steps`` and ``batch_size`` to see if you can get
   better loss!


================================================================================
Training on ST-MNIST with Tonic + snnTorch Tutorial
================================================================================

Tutorial written by Dylan Louie (djlouie@ucsc.edu), Hannah Cohen Sandler (hcohensa@ucsc.edu), Shatoparba Banerjee (sbaner12@ucsc.edu)


.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_stmnist.ipynb


The snnTorch tutorial series is based on the following paper. If you find these resources or code useful in your work, please consider citing the following source:

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  This tutorial is a static non-editable version. Interactive, editable versions are available via the following links:
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_stmnist.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


::

    pip install tonic
    pip install snntorch

::

    # tonic imports
    import tonic
    import tonic.transforms as transforms  # Not to be mistaken with torchdata.transfroms
    from tonic import DiskCachedDataset
    
    # torch imports
    import torch
    from torch.utils.data import random_split
    from torch.utils.data import DataLoader
    import torchvision
    import torch.nn as nn
    
    # snntorch imports
    import snntorch as snn
    from snntorch import surrogate
    import snntorch.spikeplot as splt
    from snntorch import functional as SF
    from snntorch import utils
    
    # other imports
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from IPython.display import display
    import numpy as np
    import torchdata
    import os
    from ipywidgets import IntProgress
    import time
    import statistics


1. The ST-MNIST Dataset
^^^^^^^^^^^^^^^^^^^^^^^

1.1 Introduction
^^^^^^^^^^^^^^^^

The Spiking Tactile-MNIST (ST-MNIST) dataset features handwritten digits
(0-9) inscribed by 23 individuals on a 100-taxel biomimetic event-based
tactile sensor array. This dataset captures the dynamic pressure changes
associated with natural writing. The tactile sensing system,
Asynchronously Coded Electronic Skin (ACES), emulates the human
peripheral nervous system, transmitting fast-adapting (FA) responses as
asynchronous electrical events.

More information about the ST-MNIST dataset can be found in the
following paper:


    `H. H. See, B. Lim, S. Li, H. Yao, W. Cheng, H. Soh, and B. C. K. Tee, “ST-MNIST - The Spiking Tactile-MNIST Neuromorphic Dataset”. 
    A PREPRINT, May 2020. [Online]. Available: <https://arxiv.org/abs/2005.04319>`_


1.2 Downloading the ST-MNIST dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


ST-MNIST is in the ``MAT`` format. Tonic can be used transform this into
an event-based format (x, y, t, p).

1. Download the compressed dataset by accessing:
   `<https://scholarbank.nus.edu.sg/bitstream/10635/168106/2/STMNIST%20dataset%20NUS%20Tee%20Research%20Group.zip>`_

2. The zip file is ``STMNIST dataset NUS Tee Research Group``. Create a
   parent folder titled ``STMNIST`` and place the zip file inside.

3. If running in a cloud-based environment, e.g., on Colab, you will
   need to do this in Google Drive.

1.3 Mount to Drive
^^^^^^^^^^^^^^^^^^

You may need to authorize the following to access Google Drive:

::

    # Load the Drive helper and mount
    from google.colab import drive
    drive.mount('/content/drive')

After executing the cell above, Drive files will be present in
“/content/drive/MyDrive”. You may need to change the ``root`` file to
your own path.

::

    root = "/content/drive/My Drive/"  # similar to os.path.join('content', 'drive', 'My Drive')
    os.listdir(os.path.join(root, 'STMNIST')) # confirm the file exists

1.4 ST-MNIST Using Tonic
^^^^^^^^^^^^^^^^^^^^^^^^

``Tonic`` will be used to convert the dataset into a format compatible
with PyTorch/snnTorch. The documentation can be found
`here <https://tonic.readthedocs.io/en/latest/generated/tonic.prototype.datasets.STMNIST.html#tonic.prototype.datasets.STMNIST>`__.

::

    dataset = tonic.prototype.datasets.STMNIST(root=root, keep_compressed = False, shuffle = False)

Tonic formats the STMNIST dataset into ``(x, y, t, p)`` tuples. 

* ``x`` is the position on the x-axis 
* ``y`` is the position on the y-axis 
* ``t`` is a timestamp 
* ``p`` is polarity; +1 if taxel pressed down, 0 if taxel released

Each sample also contains the label, which is an integer 0-9 that
corresponds to what digit is being drawn.

An example of one of the events is shown below:

::

    events, target = next(iter(dataset))
    print(events[0])
    print(target)

:: 

    >>> (2, 7, 199838, 0)
    >>> 6

The ``.ToFrame()`` function from ``tonic.transforms`` transforms events
from an (x, y, t, p) tuple to a numpy array matrix.

::

    sensor_size = tuple(tonic.prototype.datasets.STMNIST.sensor_size.values())  # The sensor size for STMNIST is (10, 10, 2)
    
    # filter noisy pixels and integrate events into 1ms frames
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                          transforms.ToFrame(sensor_size=sensor_size,
                                                             time_window=20000)
                                         ])
    
    transformed_events = frame_transform(events)
    
    print_frame(transformed_events)

::

    >>> 
    ----------------------------
    [[[0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 3 4 0 0 0 0 0 0 0]
    [0 2 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]]

    [[0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 4 0 0 0 0 0 0 0]
    [0 6 3 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]]]
    ----------------------------
    [[0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 3 4 0 0 0 0 0 0 0]
    [0 2 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]]
    ----------------------------
    [0 0 0 0 0 0 0 0 0 0]


1.5 Visualizations
^^^^^^^^^^^^^^^^^^

Using ``tonic.utils.plot_animation``, the frame transform, and also some
rotation. We can create an animation of the data and visualize this.

::

    # Iterate to a new iteration
    events, target = next(iter(dataset))

::

    frame_transform_tonic_visual = tonic.transforms.ToFrame(
        sensor_size=(10, 10, 2),
        time_window=10000,
    )
    
    frames = frame_transform_tonic_visual(events)
    frames = frames / np.max(frames)
    frames = np.rot90(frames, k=-1, axes=(2, 3))
    frames = np.flip(frames, axis=3)
    
    # Print out the Target
    print('Animation of ST-MNIST')
    print('The target label is:',target)
    animation = tonic.utils.plot_animation(frames)
    
    # Display the animation inline in a Jupyter notebook
    HTML(animation.to_jshtml())

We can also use ``snntorch.spikeplot``

::

    frame_transform_snntorch_visual = tonic.transforms.ToFrame(
        sensor_size=(10, 10, 2),
        time_window=8000,
    )
    
    tran = frame_transform_snntorch_visual(events)
    tran = np.rot90(tran, k=-1, axes=(2, 3))
    tran = np.flip(tran, axis=3)
    tran = torch.from_numpy(tran)
    
    tensor1 = tran[:, 0:1, :, :]
    tensor2 = tran[:, 1:2, :, :]
    
    print('Animation of ST-MNIST')
    print('The target label is:',target)
    
    fig, ax = plt.subplots()
    time_steps = tensor1.size(0)
    tensor1_plot = tensor1.reshape(time_steps, 10, 10)
    anim = splt.animator(tensor1_plot, fig, ax, interval=10)
    
    display(HTML(anim.to_html5_video()))

::

    >>> Animation of ST-MNIST
    >>> The target label is: 3
    

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/stmnist/stmnist-anim.mp4?raw=true"></video>
  </center>


There is a total of 6,953 recordings in this dataset. The developers of
ST-MNIST invited 23 participants to write each 10 digit approx. 30 times
each: 23*30*10 = 6,900.

::

    print(len(dataset))

::

    >>> 6953

1.6 Lets create a trainset and testset!
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ST-MNIST isn’t already seperated into a trainset and testset in Tonic.
That means we will have to seperate it manually. In the process of
seperating the data we will transform them using ``.ToFrame()`` as well.

::

    sensor_size = tonic.prototype.datasets.STMNIST.sensor_size
    sensor_size = tuple(sensor_size.values())
    
    # Define a transform
    frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size, time_window=20000)])

The following code reads out the a portion of the dataset, transforms
the events using ``frame_transform`` defined above, and then seperates
the data into a trainset and a testset. On top of that, ``.ToFrame()``
is applied each time. Thus, this code snippet might take a few minutes.

For speed, we will just use a subset of the dataset. By default, 640
training samples and 320 testing samples. Feel free to change this if
you have more patience than us.

::

    def shorter_transform_STMNIST(data, transform):
        short_train_size = 640
        short_test_size = 320
    
        train_bar = IntProgress(min=0, max=short_train_size)
        test_bar = IntProgress(min=0, max=short_test_size)
    
        testset = []
        trainset = []
    
        print('Porting over and transforming the trainset.')
        display(train_bar)
        for _ in range(short_train_size):
            events, target = next(iter(dataset))
            events = transform(events)
            trainset.append((events, target))
            train_bar.value += 1
        print('Porting over and transforming the testset.')
        display(test_bar)
        for _ in range(short_test_size):
            events, target = next(iter(dataset))
            events = transform(events)
            testset.append((events, target))
            test_bar.value += 1
    
        return (trainset, testset)
    
    start_time = time.time()
    trainset, testset = shorter_transform_STMNIST(dataset, frame_transform)
    elapsed_time = time.time() - start_time
    
    # Convert elapsed time to minutes, seconds, and milliseconds
    minutes, seconds = divmod(elapsed_time, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = round(milliseconds * 1000)
    
    # Print the elapsed time
    print(f"Elapsed time: {int(minutes)} minutes, {int(seconds)} seconds, {milliseconds} milliseconds")

1.6 Dataloading and Batching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # Create a DataLoader
    dataloader = DataLoader(trainset, batch_size=32, shuffle=True)

For faster dataloading, we can use ``DiskCashedDataset(...)`` from
Tonic.

Due to variations in the lengths of event recordings,
``tonic.collation.PadTensors()`` will be used to prevent irregular
tensor shapes. Shorter recordings are padded, ensuring uniform
dimensions across all samples in a batch.

::

    transform = tonic.transforms.Compose([torch.from_numpy])
    
    cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/stmnist/train')
    
    # no augmentations for the testset
    cached_testset = DiskCachedDataset(testset, cache_path='./cache/stmnist/test')
    
    batch_size = 32
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

::

    # Query the shape of a sample: time x batch x dimensions
    data_tensor, targets = next(iter(trainloader))
    print(data_tensor.shape)

::

    >>> torch.Size([89, 32, 2, 10, 10])

1.7 Create the Spiking Convolutional Neural Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below we have by default a spiking convolutional neural network with the
architecture: ``10×10-32c4-64c3-MaxPool2d(2)-10o``.

::

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # neuron and simulation parameters
    beta = 0.95
    
    # This is the same architecture that was used in the STMNIST Paper
    scnn_net = nn.Sequential(
        nn.Conv2d(2, 32, kernel_size=4),
        snn.Leaky(beta=beta, init_hidden=True),
        nn.Conv2d(32, 64, kernel_size=3),
        snn.Leaky(beta=beta, init_hidden=True),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 2 * 2, 10),  # Increased size of the linear layer
        snn.Leaky(beta=beta, init_hidden=True, output=True)
    ).to(device)
    
    optimizer = torch.optim.Adam(scnn_net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

1.8 Define the Forward Pass
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    def forward_pass(net, data):
        spk_rec = []
        utils.reset(net)  # resets hidden states for all LIF neurons in net
    
        for step in range(data.size(0)):  # data.size(0) = number of time steps
    
            spk_out, mem_out = net(data[step])
            spk_rec.append(spk_out)
    
        return torch.stack(spk_rec)

1.9 Create and Run the Training Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This might take a while, so kick back, take a break and eat a snack
while this happens; perhaps even count kangaroos to take a nap or do a
shoey and get schwasted instead.

::

    start_time = time.time()
    
    num_epochs = 30
    
    loss_hist = []
    acc_hist = []
    
    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)
    
            scnn_net.train()
            spk_rec = forward_pass(scnn_net, data)
            loss_val = loss_fn(spk_rec, targets)
    
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
    
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
    
            # Print loss every 4 iterations
            if i%4 == 0:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
    
            # Calculate accuracy rate and then append it to accuracy history
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
    
            # Print accuracy every 4 iterations
            if i%4 == 0:
                print(f"Accuracy: {acc * 100:.2f}%\n")
    
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    
    # Convert elapsed time to minutes, seconds, and milliseconds
    minutes, seconds = divmod(elapsed_time, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = round(milliseconds * 1000)
    
    # Print the elapsed time
    print(f"Elapsed time: {int(minutes)} minutes, {int(seconds)} seconds, {milliseconds} milliseconds")

::

    Epoch 0, Iteration 0 
    Train Loss: 8.06
    Accuracy: 9.38%

    Epoch 0, Iteration 4 
    Train Loss: 42.37
    Accuracy: 6.25%

    Epoch 0, Iteration 8 
    Train Loss: 7.07
    Accuracy: 15.62%

    Epoch 0, Iteration 12 
    Train Loss: 8.73
    Accuracy: 12.50%

    ...

    Epoch 29, Iteration 8 
    Train Loss: 0.93
    Accuracy: 100.00%

    Epoch 29, Iteration 12 
    Train Loss: 0.97
    Accuracy: 100.00%

    Epoch 29, Iteration 16 
    Train Loss: 1.38
    Accuracy: 87.50%

    Elapsed time: 2 minutes, 45 seconds, 187 milliseconds


Uncomment the code below if you want to save the model

::

    # torch.save(scnn_net.state_dict(), 'scnn_net.pth')

2. Results
^^^^^^^^^^

2.1 Plot accuracy history
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # Plot Loss
    fig = plt.figure(facecolor="w")
    plt.plot(acc_hist)
    plt.title("Train Set Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/stmnist/train-acc.png?raw=true 


2.2 Evaluate the Network on the Test Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # Make sure your model is in evaluation mode
    scnn_net.eval()
    
    # Initialize variables to store predictions and ground truth labels
    acc_hist = []
    
    # Iterate over batches in the testloader
    with torch.no_grad():
        for data, targets in testloader:
            # Move data and targets to the device (GPU or CPU)
            data = data.to(device)
            targets = targets.to(device)
    
            # Forward pass
            spk_rec = forward_pass(scnn_net, data)
    
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
    
            # if i%10 == 0:
            # print(f"Accuracy: {acc * 100:.2f}%\n")
    
    print("The average loss across the testloader is:", statistics.mean(acc_hist))

::

    >>> The average loss across the testloader is: 0.65


2.3 Visualize Spike Recordings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following visual is a spike count histogram for a single target and
single piece of data using the spike recording list.

::

    spk_rec = forward_pass(scnn_net, data)

::

    # Change index to visualize a different sample
    idx = 0
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
    print(f"The target label is: {targets[idx]}")
    
    #  Plot spike count histogram
    anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels,
                            animate=True, interpolate=1)
    
    display(HTML(anim.to_html5_video()))
    # anim.save("spike_bar.mp4")

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/stmnist/spike-count.mp4?raw=true"></video>
  </center>


Congratulations!

You trained a Spiking CNN using ``snnTorch`` and ``Tonic`` on ST-MNIST!


The tutorial consists of a series of Google Colab notebooks. Static non-editable versions are also available. 


.. list-table::
   :widths: 20 60 30
   :header-rows: 1

   * - Tutorial
     - Title
     - Colab Link
   * - `Tutorial 1 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html>`_
     - Spike Encoding with snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb

   * - `Tutorial 2 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html>`_
     - The Leaky Integrate and Fire Neuron
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb

   * - `Tutorial 3 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html>`_
     -  A Feedforward Spiking Neural Network
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb


   * - `Tutorial 4 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html>`_
     -  2nd Order Spiking Neuron Models (Optional)
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_4_advanced_neurons.ipynb

  
   * - `Tutorial 5 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html>`_
     -  Training Spiking Neural Networks with snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb
   
   * - `Tutorial 6 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html>`_
     - Surrogate Gradient Descent in a Convolutional SNN
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_6_CNN.ipynb

   * - `Tutorial 7 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html>`_
     - Neuromorphic Datasets with Tonic + snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb


.. list-table::
   :widths: 70 32
   :header-rows: 1

   * - Advanced Tutorials
     - Colab Link

   * - `Population Coding <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_pop.ipynb

   * - `Regression: Part I - Membrane Potential Learning with LIF Neurons <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb

   * - `Regression: Part II - Regression-based Classification with Recurrent LIF Neurons <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_2.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb

   * - `Binarized Spiking Neural Networks: Erik Mercado <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_binarized_spiking_neural_networks.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_BSNN.ipynb


   * - `Accelerating snnTorch on IPUs <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_ipu_1.html>`_
     -       —

.. list-table::
   :widths: 70 32
   :header-rows: 1

   * - Dataset Tutorials
     - Colab Link


   * - `Exoplanet Hunter: Finding Planets Using Light Intensity <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_exoplanet_hunter.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_exoplanet_hunter.ipynb

    
   * - `Spiking-Tactile MNIST Neuromorphic Dataset <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_stmnist.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_stmnist.ipynb
        

Future tutorials on spiking neurons and training are under development. 

