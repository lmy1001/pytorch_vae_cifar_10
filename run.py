import torch
import numpy as np
import cifar10_data
import vae
import plot_utils
import random
import argparse
from tensorboardX import SummaryWriter
import vae2

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--type', type=str, default='ae',
                        help='use autoencoder or variational autoencoder', choices=['ae', 'vae'])
    parser.add_argument('--data_dir', type=str,
                        default='/Users/lmy/Dataset/Cifar-10/cifar-10-batches-py/',
                        #default='/srv/beegfs-benderdata/scratch/density_estimation/data/3DMultiView/cifar-10-batches-py/',
                        help='directory of dataset')
    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')
    parser.add_argument('--train_log_name', type=str, default='log/ae/train_log',
                        help='File name of train log event')
    parser.add_argument('--val_log_name', type=str, default='log/ae/val_log',
                        help='File name of validation log event')
    parser.add_argument('--use_batch_norm', type=bool, default=True,
                        help='Boolean for using batch normalization')

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer to choose', choices=['adam', 'sgd'])

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')

    parser.add_argument('--zdim', type=int, default=64, help='Dimension of latent vector')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=120, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for initializing training. ')

    parser.add_argument('--show_n_img_x', type=int, default=3,
                        help='Number of images along x-axis')

    parser.add_argument('--show_n_img_y', type=int, default=3,
                        help='Number of images along y-axis')

    parser.add_argument('--show_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    return parser.parse_args()


def main(args):
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    '''prepare cifar 10 data '''
    train_data, train_label, val_data, val_label, test_data, test_labels = cifar10_data.prepare_cifar_10_data()

    n_train_samples = train_data.shape[0]
    n_samples = n_train_samples         #only train 2000 images as test
    #n_samples = 2000
    n_val_samples = val_data.shape[0]
    n_test_samples = test_data.shape[0]
    image_size = 32
    input_dim = 3
    output_dim = 3
    train_data = train_data.reshape(n_train_samples, input_dim, image_size, image_size)
    train_data = train_data[:n_samples, :, :, :]
    val_data = val_data.reshape(n_val_samples, input_dim, image_size, image_size)
    test_data = test_data.reshape(n_test_samples, input_dim, image_size, image_size)

    """ create network """
    encoder = vae2.Encoder(input_dim, args.zdim, args.use_batch_norm).to(device)
    decoder = vae2.Decoder(args.zdim, output_dim, args.use_batch_norm).to(device)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learn_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learn_rate, momentum=args.momentum)
    else:
        print("wrong optimizer")
        return

    # init weights
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    cifar10_data.set_random_seed(args.seed)

    #output log to writer
    writer_train = SummaryWriter(logdir=args.train_log_name)
    writer_val = SummaryWriter(logdir=args.val_log_name)
    mode = args.type

    # Plot for reproduce performance
    image_size = 32
    plot_perform = plot_utils.Plot_Reproduce_Performance(args.results_path, args.show_n_img_x, args.show_n_img_y, image_size,
                                                    image_size, args.show_resize_factor)


    show_img_nums = args.show_n_img_x * args.show_n_img_y
    show_img = val_data[:show_img_nums, :, :, :]
    #show_img = val_data.data[:show_img_nums]
    print(show_img.shape)           # N * 32 * 32 *3
    plot_perform.save_images(show_img, name='input.jpg')
    print('saved:', 'input.jpg')

    """ training """
    batch_size = args.batch_size
    epochs = args.num_epochs
    num_batches = int(n_samples / batch_size)
    for epoch in range(epochs):

        encoder.train()
        decoder.train()
        tot_loss = 0
        tot_l2_loss = 0
        for i in range(num_batches):
            idx = (i * batch_size) % (n_samples)
            end_idx =  idx + batch_size
            batch_input = train_data[idx:end_idx, :, :, :]
            batch_target = batch_input
            batch_input, batch_target = torch.from_numpy(batch_input).float().to(device), \
                                        torch.from_numpy(batch_target).float().to(device)

            y, z, loss, loss_likelihood, loss_divergence, l2_dis = \
                vae2.get_loss(encoder, decoder, batch_input, batch_target, mode)

            tot_loss = tot_loss + loss.item()
            tot_l2_loss = tot_l2_loss + l2_dis.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print cost every epoch
            if i == num_batches - 1:
                print("train loss: epoch %d: Loss %03.2f L_likelihood %03.2f L_divergence %03.2f L2_dis %03.2f " % (
                    epoch, loss.item(), loss_likelihood.item(), loss_divergence.item(), l2_dis.item()))
                break

        tot_loss = tot_loss / float(num_batches)
        tot_l2_loss = tot_l2_loss / float(num_batches)
        writer_train.add_scalar('loss', tot_loss, epoch)
        writer_train.add_scalar('l2_dis', tot_l2_loss, epoch)

        #evaluate on val dataset

        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            encoder.eval()
            decoder.eval()

            with torch.no_grad():
                #calculate the validation loss
                val_target = val_data
                val_input = val_data
                val_input, val_target = torch.from_numpy(val_input).float().to(device), \
                                        torch.from_numpy(val_target).float().to(device)

                y_val, z_val, loss_val, loss_likelihood_val, loss_divergence_val, l2_dis_val = \
                                vae2.get_loss(encoder, decoder, val_input, val_target, mode)
                print("test results in val data: epoch %d: Loss %03.2f L_likelihood %03.2f L_divergence %03.2f L2_dis %03.2f " % (
                    epoch, loss_val.item(), loss_likelihood_val.item(), loss_divergence_val.item(), l2_dis_val.item()))

            # Plot for reproduce performance
            plot_batch = torch.from_numpy(show_img).float().to(device)
            y_PRR = vae2.get_ae(encoder, decoder, plot_batch)
            y_PRR_img = y_PRR.reshape(show_img_nums, 3, image_size, image_size)
            print(y_PRR_img.shape)
            plot_perform.save_images(y_PRR_img.detach().cpu().numpy(), name="/PRR_epoch_%02d" % (epoch) + ".jpg")
            print('saved:', "/PRR_epoch_%02d" % (epoch) + ".jpg")
            writer_val.add_scalar('loss', loss_val, epoch)
            writer_val.add_scalar('l2_dis', l2_dis_val, epoch)

    writer_train.close()
    writer_val.close()


    plot_utils.plot_t_sne(z_val[:100, :].detach().cpu().numpy(), val_data[:100, :, :, :])
    plot_utils.plot_t_sne_col(z_val.detach().cpu().numpy(), val_label)

if __name__=='__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)
