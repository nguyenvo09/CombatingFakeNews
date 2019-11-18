import sys
sys.path.insert(0,'../../pytorch')
import os
from Models import GAU as gau
import argparse
import time
import interactions
from handlers import data_handlers as my_data_handlers


def fit_models(args):
    train_file = os.path.join(args.path, args.dataset, '%s.train.rating' % args.dataset)
    vad_file = os.path.join(args.path, args.dataset, '%s.dev.rating' % args.dataset)
    test_file = os.path.join(args.path, args.dataset, '%s.test.rating' % args.dataset)

    network_file = os.path.join(args.path, args.dataset, '%s.adjacency.network' % args.dataset)
    user_user_sppmi_raw_file = os.path.join(args.path, args.dataset, '%s.user_user.frequency.csv' % args.dataset)
    item_item_sppmi_raw_file = os.path.join(args.path, args.dataset, '%s.item_item.frequency.csv' % args.dataset)
    user_user_sim_file = os.path.join(args.path, args.dataset, '%s.user_user_cosine_sim.csv' % args.dataset)
    item_item_sim_file = os.path.join(args.path, args.dataset, '%s.url_url_cosine_sim.csv' % args.dataset)

    trainRatings = my_data_handlers.load_rating_file_as_dict(train_file)
    vadRatings = my_data_handlers.load_rating_file_as_dict(vad_file)
    testRatings = my_data_handlers.load_rating_file_as_dict(test_file)
    vadNegatives = my_data_handlers.generate_negatives(vadRatings, removed = [trainRatings, testRatings])
    testNegatives = my_data_handlers.generate_negatives(testRatings, removed = [trainRatings, vadRatings])

    rec_model = gau.GAU_model(loss = args.loss_type,  # 'pointwise, bpr, hinge, adaptive_hinge'
                              embedding_dim = args.num_factors,
                              n_iter = args.epochs,
                              batch_size = args.batch_size,
                              reg_l2 = args.reg,  # L2 regularization
                              learning_rate = args.lr,  # learning_rate
                              decay_step = args.decay_step,  # step to decay the learning rate
                              decay_weight = args.decay_weight,  # percentage to decay the learning rat.
                              optimizer_func = None,
                              use_cuda = args.cuda,
                              random_state = None,
                              num_negative_samples = args.num_neg,
                              trained_net = None,
                              net_type = args.model,
                              logfolder = args.log,
                              full_settings = args)
    # t0 = time.time()
    t1 = time.time()
    print('parsing data')
    train_iteractions = interactions.load_data(train_file, dataset = args.dataset)
    adjNetwork = my_data_handlers.load_network(network_file)
    item_item_mat_freq = my_data_handlers.load_data_item_item_for_sppmi(item_item_sppmi_raw_file, no_items = train_iteractions.num_items)
    user_user_mat_freq = my_data_handlers.load_data_user_user_for_sppmi(user_user_sppmi_raw_file, no_guardians = train_iteractions.num_users)
    item_item_sppmi = my_data_handlers.computeSPPMI_matrix(item_item_mat_freq, shifted_k = args.shifted_k)
    user_user_sppmi = my_data_handlers.computeSPPMI_matrix(user_user_mat_freq, shifted_k = args.shifted_k)
    user_user_sim = my_data_handlers.load_sim(user_user_sim_file, train_iteractions.num_users)
    item_item_sim = my_data_handlers.load_sim(item_item_sim_file, train_iteractions.num_items)
    print('done extracting')
    t2 = time.time()
    print('loading data time: %d (seconds)' % (t2 - t1))

    print('building the model')

    try:
        rec_model.fit(train_iteractions,
                      verbose = True,  # for printing out evaluation during training
                      topN = 10,
                      vadRatings = vadRatings, vadNegatives = vadNegatives,
                      testRatings = testRatings, testNegatives = testNegatives,
                      adjNetwork = adjNetwork,
                      user_user_sppmi = user_user_sppmi,
                      item_item_sppmi = item_item_sppmi,
                      user_user_sim = user_user_sim,
                      item_item_sim = item_item_sim,
                      alpha_gau = args.alpha_gau,
                      gamma_gau = args.gamma_gau,
                      beta_gau = args.beta_gau)
    except KeyboardInterrupt:
        print('Exiting from training early')
    t10 = time.time()
    print('Total time:  %d (seconds)' % (t10 - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Description: GAU Model")
    parser.add_argument('--path', default = '../Splitted_data', help = 'Input data path', type = str)
    parser.add_argument('--dataset', default = 'sigir18', help = 'Dataset types', type = str)
    parser.add_argument('--epochs', default = 100, help = 'Number of epochs to run', type = int)
    parser.add_argument('--batch_size', default = 256, help = 'Batch size', type = int)
    parser.add_argument('--num_factors', default = 8, help = 'number of latent factors', type = int)
    parser.add_argument('--reg', type=float, default = 1e-6, help = 'Regularization for users and item embeddings')
    parser.add_argument('--num_neg', default = 3, type = int, help = 'Number of negative instances for each positive sample')
    parser.add_argument('--shifted_k', default = 2, type = int, help = 'shifted_k for computing SPPMI matrices: [1, 2, 5, 10, 50]')
    parser.add_argument('--alpha_gau', default = 0.8, type = float, help = 'factor to control contribution of network information')
    parser.add_argument('--gamma_gau', default = 0.8, type = float, help = 'factor to control contribution of user similarity')
    parser.add_argument('--beta_gau', default = 0.8, type = float, help = 'factor to control contribution of item similarity')
    parser.add_argument('--lr', default = 0.001, type = float, help = 'Learning rate')
    parser.add_argument('--log', default = "../logs/GAU_log", type = str, help = 'folder for logs and saved models')
    parser.add_argument('--optimizer', nargs = '?', default = 'adam', help = 'Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--loss_type', nargs = '?', default = 'single_pointwise_square_loss', help = 'Specify a loss function: bce, pointwise, bpr, hinge, adaptive_hinge, single_pointwise_square_loss')
    parser.add_argument('--model', default = 'gau', help = 'Selecting the model type [gau]', type = str)
    parser.add_argument('--topk', type = int, default = 10, help = 'top K')
    parser.add_argument('--cuda', type = int, default = 1, help = 'using cuda or not')
    parser.add_argument('--decay_step', type = int, default = 100, help = 'how many steps to decay the learning rate')
    parser.add_argument('--decay_weight', type = float, default = 0.0001, help = 'percent of decaying')

    args = parser.parse_args()
    fit_models(args)

