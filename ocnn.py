"""
Python script for outlier detection based on One-Class Neural Network (OC-NN).

Copyright (C) 2021 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import OrderedDict

import numpy as np
import torch
from pyod.models.base import BaseDetector
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
from tqdm import tqdm


class PyODDataset(torch.utils.data.Dataset):
    """
    PyOD Dataset class for PyTorch Dataloader.
    """

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if type(self.mean) is np.ndarray:
            sample = (sample - self.mean) / self.std

        return torch.from_numpy(sample), idx


class AutoEncoder(nn.Module):
    """
    Autoencoder for pretraining.
    """

    def __init__(
        self,
        n_features,
        hidden_neurons=[128, 64],
        dropout_rate=0.0,
        batch_norm=False,
        hidden_activation=None,
    ):
        super(AutoEncoder, self).__init__()
        self.hidden_activation = hidden_activation
        self.hidden_neurons = hidden_neurons

        # Build encoder
        modules = []
        in_features = n_features
        for hidden_neurons in self.hidden_neurons:
            modules.append(
                nn.Linear(in_features, out_features=hidden_neurons, bias=False)
            )
            if batch_norm is True:
                modules.append(nn.BatchNorm1d(hidden_neurons))

            if self.hidden_activation is not None:
                modules.append(self.hidden_activation)
            in_features = hidden_neurons
        self.encoder = nn.Sequential(*modules)

        # Build decoder
        modules = []
        reversed_neuron_list = list(reversed(self.hidden_neurons))
        in_features = list(reversed_neuron_list)[0]
        for reversed_neurons in reversed_neuron_list[1:]:
            modules.append(
                nn.Linear(
                    in_features, out_features=reversed_neurons, bias=False
                )
            )
            if batch_norm is True:
                modules.append(nn.BatchNorm1d(reversed_neurons))

            if self.hidden_activation is not None:
                modules.append(self.hidden_activation)
            in_features = reversed_neurons

        modules.append(
            nn.Linear(
                in_features=self.hidden_neurons[0],
                out_features=n_features,
                bias=False,
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        """
        forward.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class OneClassNN_net(nn.Module):
    """
    Network for OneClassNN.
    """

    def __init__(
        self,
        n_features,
        hidden_neurons=[16, 8],
        ocnn_neurons=[8, 1],
        dropout_rate=0.0,
        batch_norm=False,
        hidden_activation=None,
        output_activation=None,
    ):
        super(OneClassNN_net, self).__init__()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_neurons = hidden_neurons

        # Build encoder
        modules = []
        in_features = n_features
        for hidden_neurons in self.hidden_neurons:
            modules.append(
                nn.Linear(in_features, out_features=hidden_neurons, bias=False)
            )
            if batch_norm is True:
                modules.append(nn.BatchNorm1d(hidden_neurons))

            if self.hidden_activation is not None:
                modules.append(self.hidden_activation)

            in_features = hidden_neurons
        self.encoder = nn.Sequential(*modules)

        # Build a feed-forward NN
        modules = OrderedDict([])
        modules["input_to_hidden"] = nn.Linear(
            in_features=self.hidden_neurons[-1],
            out_features=ocnn_neurons[0],
            bias=False,
        )
        self.V = modules["input_to_hidden"].weight
        if batch_norm is True:
            modules["batchnorm1"] = nn.BatchNorm1d(ocnn_neurons[0])

        if self.output_activation is not None:
            modules["activation"] = self.output_activation

        modules["hidden_to_output"] = nn.Linear(
            in_features=ocnn_neurons[0],
            out_features=ocnn_neurons[1],
            bias=False,
        )
        self.w = modules["hidden_to_output"].weight
        if batch_norm is True:
            modules["batchnorm2"] = nn.BatchNorm1d(ocnn_neurons[1])

        self.ocnn_ff = nn.Sequential(modules)

    def forward(self, inputs):
        """
        Perform forward propagation.
        """
        hidden = self.encoder(inputs)
        outputs = self.ocnn_ff(hidden)

        return outputs


class OneClassNN(BaseDetector):
    """One-Class Neural Network (OC-NN) is a type of neural networks
    which learns useful data representations unsupervisedly for
    anomaly/outlier detection. OC-NN can be positioned as an extension of
    OC-SVM using deep learning.
    For details, see https://arxiv.org/abs/1802.06360.

    Parameters
    ----------
    nu: float, optional (default=0.1)
        One-Class NN hyperparameter nu (must be 0 < nu <= 1).

    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. if use_reconst is True, neurons
        will be reversed eg. [64, 32] -> [64, 32, 32, 64, n_features]

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        If this is set to None, linear activation is applied (identical function).

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        If this is set to None, linear activation is applied (identical function).

    ocnn_neurons : list, optional (default=[32, 1])
        The number of neurons for OC-NN layers.
        The neurons at the last layer must be 1.

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    leaning_rate : float in (0., 1), optional (default=0.001)
        Learning rate to be used in updating network weights.

    weight_decay : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer.

    validation_size : float in (0., 1), optional (default=0.0)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=False)
        If True, apply standardization on the data.

    pretraining : bool, optional (default=False)
        If True, an autoencoder is pretrained.
        The network weights of OC-NN will be initialized
        by using pre-trained weights from the autoencoder.

    pretrain_epochs : int, optional (default=10)
        Number of epochs to pre-train the autoencoder.

    batch_norm : bool, optional (default=False)
        If True, apply standardization on the data.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    criterion : Torch Module, optional (default=torch.nn.MSEloss)
        A criterion that measures erros between
        network output from autoencoder and input.
        It should be noticed that this is used only for pre-training.

    verbose : int, optional (default=0)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(
        self,
        r=0.0,
        nu=0.1,
        hidden_neurons=[64, 32],
        hidden_activation="relu",
        ocnn_neurons=[32, 1],
        batch_norm=False,
        output_activation="sigmoid",
        pretraining=False,
        pretrain_epochs=10,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        dropout_rate=0.0,
        validation_size=0.0,
        preprocessing=False,
        criterion=torch.nn.MSELoss(),
        verbose=0,
        contamination=0.1,
        device=None,
    ):
        super(OneClassNN, self).__init__(contamination=contamination)

        assert (0 < nu) & (
            nu <= 1
        ), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.r = torch.tensor(r, device=self.device)

        self.hidden_neurons = hidden_neurons
        self.hidden_activation = self._get_activation_by_name(hidden_activation)
        self.output_activation = self._get_activation_by_name(output_activation)

        assert (
            ocnn_neurons[-1] == 1
        ), "The number of neurons at the last layer in OC-NN must be 1."
        self.ocnn_neurons = ocnn_neurons

        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.pretraining = pretraining
        self.pretrain_epochs = pretrain_epochs
        self.dropout_rate = dropout_rate
        self.validation_size = validation_size
        self.batch_norm = batch_norm
        self.criterion = criterion
        self.verbose = verbose

    def _get_activation_by_name(self, name):
        """
        Get activation function by name (string).
        """

        activations = {
            "relu": nn.ReLU(),
            "prelu": nn.PReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "linear": None,
        }

        if name in activations.keys():
            return activations[name]

        else:
            raise ValueError(name, "is not a valid activation function")

    def _update_bias(self, y_pred, nu):
        """
        Optimally solve for bias r via the nu-quantile of distances.
        """
        new_nu = np.quantile(y_pred.cpu().detach().numpy(), q=nu)
        self.r.data = torch.tensor(new_nu, device=self.device)

    def _train_AutoEncoder(self, train_loader):
        """
        Internal function to train AutoEncoder.

        Parameters
        ----------
        train_loader : torch dataloader
            Data loader of training data.
        """

        optimizer = torch.optim.Adam(
            self.ae_net.parameters(), lr=self.learning_rate
        )

        self.ae_net.train()
        for epoch in range(self.pretrain_epochs):
            training_loss = 0.0
            for data, _ in train_loader:
                inputs = data.to(self.device).float()
                optimizer.zero_grad()

                outputs = self.ae_net(inputs)
                loss = self.criterion(inputs, outputs)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

    def _init_network_weights_from_pretraining(self):
        """
        Initialize the OCNN network weights from the encoder
        weights of the pretraining autoencoder.
        """

        net_dict = self.ocnn_net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}

        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)

        # Load the new state_dict
        self.ocnn_net.load_state_dict(net_dict)

    def _train_OneClassNN(self, train_loader, val_loader):
        """
        Internal function to train OneClassNN.

        Parameters
        ----------
        train_loader : torch dataloader
            Data loader of training data.

        val_loader : torch dataloader
            Data loader of validation data.
        """

        optimizer = torch.optim.AdamW(
            self.ocnn_net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        tqdm_disable = True
        if self.verbose == 1:
            tqdm_disable = False

        for epoch in tqdm(range(self.epochs), disable=tqdm_disable):
            training_loss = []
            self.ocnn_net.train()
            for data, _ in train_loader:
                inputs = data.to(self.device).float()
                optimizer.zero_grad()

                outputs = self.ocnn_net(inputs)  # hat{y}
                diff = self.r - outputs  # y - hat{y} in eq.(4)

                loss = (1 / self.nu) * torch.mean(
                    torch.max(torch.zeros_like(diff), diff)
                )

                # add loss from w and V
                # w: weights from hidden to output layer
                # V: weights from input to hidden layer
                loss_weight = (
                    torch.norm(self.ocnn_net.w) ** 2
                    + torch.norm(self.ocnn_net.V) ** 2
                )

                loss = loss + 0.5 * loss_weight

                # update weights w and V via backpropagation
                loss.backward()
                optimizer.step()

                # update bias
                self._update_bias(outputs, self.nu)

                # record objective function
                loss = loss - self.r
                training_loss.append(loss.item())


            # print(abs(float(loss_weight) * 0.5 / (float(loss) / len(train_loader))))
            # nu_value  =  abs(float(loss_weight) / float(loss) * float(self.nu))
            # print('nu = ', self.nu)

            if len(val_loader) > 0:
                self.ocnn_net.eval()
                val_loss = []
                with torch.no_grad():
                    for data, _ in val_loader:
                        inputs = data.to(self.device).float()
                        outputs = self.ocnn_net(inputs)  # hat{y}
                        diff = self.r - outputs  # y - hat{y} in eq.(4)
                        loss = (1 / self.nu) * torch.mean(
                            torch.max(torch.zeros_like(diff), diff)
                        )
                        loss_weight = (
                            torch.norm(self.ocnn_net.w) ** 2
                            + torch.norm(self.ocnn_net.V) ** 2
                        )
                        loss = loss + 0.5 * loss_weight - self.r
                        val_loss.append(loss.item())

            if len(val_loader) > 0 and self.verbose == 2:
                print(
                    "Epoch {}/{}: train_loss={:.6f}, val_loss={:.6f}".format(
                        epoch + 1,
                        self.epochs,
                        np.mean(training_loss),
                        np.mean(val_loss),
                    )
                )
            elif self.verbose == 2:
                print(
                    "Epoch {}/{}: loss={:.6f}".format(
                        epoch + 1, self.epochs, np.mean(training_loss)
                    )
                )
            # print(self.nu)
            # print(len(train_loader))
            # print(abs((float(loss) / len(train_loader))  / (float(loss_weight) * 0.5 * float(self.nu))))

            # それぞれの項の値の大きさを出力
            # print('loss_weight =', float(loss_weight * 0.5))
            # print('loss =', float(loss))
            # print(abs(float(loss) / (float(loss_weight) * 0.5)))

            self.nu = abs((float(loss)  / (float(loss_weight) * 0.5 * float(self.nu))) / 300000

            # print(self.nu)


    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        n_features = X.shape[1]

        # make dataset and dataloader
        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X=X)

        train_size = int(len(dataset) * (1.0 - self.validation_size))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Initialize One-Class Neural Network (OCNN)
        self.ocnn_net = OneClassNN_net(
            n_features=n_features,
            hidden_neurons=self.hidden_neurons,
            ocnn_neurons=self.ocnn_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
        )
        self.ocnn_net = self.ocnn_net.to(self.device)

        # pre-training using autoencoder
        if self.pretraining is True:

            # initialize autoencoder
            self.ae_net = AutoEncoder(
                n_features=n_features,
                hidden_neurons=self.hidden_neurons,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                hidden_activation=self.hidden_activation,
            )
            self.ae_net = self.ae_net.to(self.device)

            # perform training
            self._train_AutoEncoder(train_loader)

            # copy weights from AE to OCNN
            self._init_network_weights_from_pretraining()

        # perform training of OCNN
        self._train_OneClassNN(train_loader, val_loader)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        outlier_scores : numpy array of shape (n_samples,)
            The outlier score of the input samples.
        """
        check_is_fitted(self, ["ocnn_net"])
        X = check_array(X)

        if self.preprocessing:
            # self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            valid_set = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            valid_set = PyODDataset(X=X)

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # enable the evaluation mode
        self.ocnn_net.eval()

        # compute outlier scores over input samples.
        outlier_scores = []
        with torch.no_grad():
            for data, data_idx in valid_loader:
                inputs = data.to(self.device).float()
                diff = self.ocnn_net(inputs) - self.r
                score = diff.to("cpu").detach().numpy().copy()
                outlier_scores.append(-1.0 * score)

        outlier_scores = np.concatenate(outlier_scores)
        return outlier_scores
