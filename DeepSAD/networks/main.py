from networks.lstm import LSTM, LSTMAutoencoder

def build_network(net_name, seq_len, n_features, embedding_dim, batch_size, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('LSTM')
    assert net_name in implemented_networks

    net = None

    if net_name == 'LSTM':
        net = LSTM(seq_len, n_features, embedding_dim, batch_size)

    return net 

def build_autoencoder(net_name, seq_len, n_features, embedding_dim, batch_size, device):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('LSTM')

    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'LSTM':
        ae_net = LSTMAutoencoder(seq_len, n_features, embedding_dim, batch_size, device)

    return ae_net 