import os
import numpy as np
import torch
import torchaudio.functional as F
import metrics
import render
import evaluate
import binauralize
import rooms.dataset
import argparse

"""
train.py is used for training and evaluation.
"""

torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def makedir_if_needed(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def initialize(indices, source_xyz, listener_xyzs, n_surfaces, load_dir):
    """
    Creates a list of ListenerLocations based on a precomputed reflections from load_dir.
    
    Parameters
    ----------
    indices: list of int (len K), of indices in the load_dir to create ListenerLocations for
    source_xyz: (3,) np.array of the source locatoin
    listener_xyzs: (N,3) np.array of the listener locations (ALL data points).
    n_surface: int representing the number of surfaces
    load_dir: directory to load precomputed reflection paths from.

    Returns
    -------
    list of ListenerLocation, corresponding to indices.
    """
    Ls = []

    for idx in indices:
        print("Loading paths from "+ load_dir)
        reflections = np.load(os.path.join(load_dir,"reflections/"+str(idx)+".npy"), allow_pickle=True)
        transmissions = np.load(os.path.join(load_dir, "transmissions/"+str(idx)+".npy"), allow_pickle=True)
        delays = np.load(os.path.join(load_dir, "delays/"+str(idx)+".npy"))
        start_directions = np.load(os.path.join(load_dir, "starts/"+str(idx)+".npy"))
        end_directions = np.load(os.path.join(load_dir, "ends/"+str(idx)+".npy"))
        L = render.ListenerLocation(source_xyz=source_xyz,
                                    listener_xyz = listener_xyzs[idx],
                                    n_surfaces=n_surfaces, reflections=reflections,
                                    transmissions=transmissions, delays=delays, 
                                    start_directions=start_directions, end_directions=end_directions)
        Ls.append(L)
    return Ls

def train_loop(R, Ls, train_gt_audio, D = None,
                n_epochs=1000, batch_size=4, lr = 1e-2, loss_fcn = None,
                save_dir=None, 
                pink_noise_supervision = False, pink_start_epoch=500,
                continue_train=False,
                fs=48000):
    """
    Runs the training process

    Parameters
    ----------
    R: Renderer
        renderer to train
    Ls: list of ListenerLocation
        training set of listener locations
    train_gt_audio: np.array(n_rirs, rir_length)
        ground truth RIRs
    save_dir: str
        path to save weights in
    perturb_surfaces: bool
        if we should perturb surfaces (and thus retrace) during training
    pink_noise_supervision: bool
        if we should supervise using pink noise during training
    pink_start_epoch: int
        what epoch we should start supervising the model on pink noise

    Returns
    -------
    losses: list of float training losses.
    """

    print("Loss:\t"+str(loss_fcn))
    print("Late Network Style\t" + R.late_stage_model)
    if save_dir is not None:
        makedir_if_needed(save_dir)

    train_gt_audio = torch.Tensor(train_gt_audio).cuda()

    # Lower learning rate on residual
    my_list = ['RIR_residual']
    my_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, R.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, R.named_parameters()))))
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': my_params, 'lr': 1e-4}], lr=lr)

    for name, param in R.named_parameters():
        print(name)

    losses = []
    
    if args.continue_train:
        losses = list(np.load(os.path.join(save_dir,"losses.npy")))
        N_train = len(Ls)
        epoch = int(len(losses)/(int(N_train)))
        print("CURRENT EPOCH")
        print(epoch)
    else:
        epoch = 0

    
    while epoch < n_epochs:

        print(epoch, flush=True)

        N_train = len(Ls)
        N_iter = max(int(N_train/batch_size),1)
        rand_idx = np.random.permutation(N_train)

        for i in range(N_iter):
            curr_indices = rand_idx[i*batch_size:(i+1)*batch_size]            
            optimizer.zero_grad()

            for idx in curr_indices:

                output = R.render_RIR(Ls[idx])
                loss = loss_fcn(output, train_gt_audio[idx])

                if pink_noise_supervision and epoch >= pink_start_epoch:

                    print("Generating Pink Noise")
                    pink_noise = generate_pink_noise(5*fs, fs=fs)
                    convolved_pred = F.fftconvolve(output, pink_noise)[...,:5*fs]
                    convolved_gt =  F.fftconvolve(train_gt_audio[idx,:R.RIR_length], pink_noise)[...,:5*fs]
                    pink_noise_loss = loss_fcn(convolved_pred, convolved_gt)
                    loss += pink_noise_loss*0.2
                
                loss.backward()
                losses.append(loss.item())
                print(loss.item(),flush=True)

            optimizer.step()

        if save_dir is not None:
            torch.save({
            'model_state_dict': R.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir,"weights.pt"))
            np.save(os.path.join(save_dir,"losses.npy"), np.array(losses))
        epoch += 1

    return losses


#Note - this function relies on precomputed reflection paths
def inference(R, source_xyz, xyzs, load_dir, source_axis_1=None, source_axis_2=None):
    """
    Render monoaural RIRs at given precomputed reflection paths.

    Parameters
    ----------
    R: Renderer
        renderer to perform inference on
    source_xyz: np.array (3,)
        3D location of source in meters
    xyzs: np.array (N, 3)
        set of listener locations to render at
    load_dir: str
        directory to load precomputed listener paths
    source_axis_1: np.array (3,)
        first axis specifying virtual source rotation,
        default is None which is (1,0,0)
    source_axis_2: np.array (3,)
        second axis specifying virtual source rotation,
        default is None which is (0,1,0)    

    Returns
    -------
    predictions: np.array (N, T) of predicted RIRs    
    """

    predictions = np.zeros((xyzs.shape[0], R.RIR_length))

    with torch.no_grad():
        R.toa_perturb = False
        for idx in range(xyzs.shape[0]):
            print(idx, flush=True)
            reflections = np.load(os.path.join(load_dir, "reflections/"+str(idx)+".npy"), allow_pickle=True)
            transmissions = np.load(os.path.join(load_dir, "transmissions/"+str(idx)+".npy"), allow_pickle=True)
            delays = np.load(os.path.join(load_dir, "delays/"+str(idx)+".npy"),allow_pickle=True)
            start_directions = np.load(os.path.join(load_dir, "starts/"+str(idx)+".npy"))

            L = render.ListenerLocation(
                source_xyz=source_xyz,
                listener_xyz=xyzs[idx],
                n_surfaces=R.n_surfaces,
                reflections=reflections,
                transmissions=transmissions,
                delays=delays,
                start_directions = start_directions)

            predict = R.render_RIR(L, source_axis_1=source_axis_1, source_axis_2=source_axis_2)
            predictions[idx] = predict.detach().cpu().numpy()

    return predictions



def generate_pink_noise(N, vol_factor = 0.04, freq_threshold=25, fs=48000):
    """
    Generates Pink Noise

    Parameters
    ----------
    N: length of audio in samples
    vol_factor: scaling factor to adjust volume to approximately match direct-line volume
    thres: frequency floor in hertz, below which the pink noise will not have any energy
    fs: sampling rate

    Returns
    -------
    pink_noise: (N,) generated pink noise
    """
    X_white = torch.fft.rfft(torch.randn(N).to(device))
    freqs = torch.fft.rfftfreq(N).to(device)
    
    normalized_freq_threshold = freq_threshold/fs
    
    pink_noise_spectrum = 1/torch.where(freqs<normalized_freq_threshold, float('inf'), torch.sqrt(freqs))
    pink_noise_spectrum = pink_noise_spectrum / torch.sqrt(torch.mean(pink_noise_spectrum**2))
    X_pink = X_white * pink_noise_spectrum
    pink_noise = torch.fft.irfft(X_pink*vol_factor)
    return pink_noise




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', help="Where to save weights/plots")
    parser.add_argument('dataset_name', help="Name of Dataset, e.g. classroomBase")
    parser.add_argument('load_dir', help="Where to load precomputed paths")

    parser.add_argument('--n_epochs', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning Rate")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch Size")

    parser.add_argument('--loss', default="training_loss", help="loss function in metrics.py")
    parser.add_argument('--continue_train', action='store_true',default=False,
                        help="continue train from checkpoint in save_dir")

    parser.add_argument('--late_stage_model', default="UniformResidual", help="Model for late stage diffuse field")
    parser.add_argument('--n_fibonacci', type=int, default=128, help="Number of Points to distribute on a sphere")    
    parser.add_argument('--toa_perturb', action='store_true', default=True, help="time-of-arrival perturbation")
    parser.add_argument('--model_transmission', action='store_true', default=False, help="Transmission Modeling")
    parser.add_argument('--fs',type=int, default=48000, help="Sample Rate")

    parser.add_argument('--pink_noise_supervision', action='store_true', default=True, help="Use pink noise")
    parser.add_argument('--pink_start_epoch', type=int, default=500, help="N. epochs before we train with pink noise")

    #Skipping various stages
    parser.add_argument('--skip_train', action='store_true',default=False, help="Skip training")
    parser.add_argument('--skip_inference', action='store_true',default=False, help="Skip rendering RIRs")
    parser.add_argument('--skip_eval', action='store_true',default=False, help="Skip evaluation")
    parser.add_argument('--skip_music', action='store_true',default=False, help="Skip rendering music")
    parser.add_argument('--skip_binaural', action='store_true',default=False, help="Skip binaural rendering")
    parser.add_argument('--valid',action='store_true', default=False, help="Evaluate on valid instead of test")
    args = parser.parse_args()

    #Loading Dataset
    print("Loading Dataset:\t" + args.dataset_name)
    D = rooms.dataset.dataLoader(args.dataset_name)

    R = render.Renderer(n_surfaces=len(D.all_surfaces), n_fibonacci=args.n_fibonacci,
                        late_stage_model=args.late_stage_model,
                        toa_perturb = args.toa_perturb, model_transmission=args.model_transmission).cuda()
    loss_fcn = getattr(metrics, args.loss) #Get loss function from metrics.py
    gt_audio = D.RIRs[:, :R.RIR_length]


    """
    Training
    """
    if not args.skip_train:
        print("Training")
        print("Loading Paths from\t" + args.load_dir)

        #Initialize Listeners
        Ls = initialize(indices=D.train_indices,
                        listener_xyzs=D.xyzs,
                        source_xyz=D.speaker_xyz,
                        n_surfaces=len(D.all_surfaces),
                        load_dir=args.load_dir)
            
        if args.continue_train:
            R.load_state_dict(torch.load(os.path.join(args.save_dir,"weights.pt"))['model_state_dict'])

        losses = train_loop(R=R, Ls=Ls, train_gt_audio=gt_audio[D.train_indices], D=D,
                            n_epochs = args.n_epochs, batch_size = args.batch_size, lr = args.lr, loss_fcn = loss_fcn,
                            save_dir=args.save_dir,
                            pink_noise_supervision = args.pink_noise_supervision,
                            pink_start_epoch=args.pink_start_epoch,
                            continue_train = args.continue_train, fs=args.fs)

    else:
        R.load_state_dict(torch.load(os.path.join(args.save_dir,"weights.pt"))['model_state_dict'])
        R.train = False
        R.toa_perturb = False

    

    """
    Inference, rendering RIRs
    """
    R.train = False
    R.toa_perturb = False
    pred_dir = os.path.join(args.save_dir, "predictions")
    if not args.skip_inference:
        pred_rirs = inference(R=R, source_xyz=D.speaker_xyz, xyzs=D.xyzs, load_dir=args.load_dir)
        makedir_if_needed(pred_dir)
        np.save(os.path.join(pred_dir, "pred_rirs.npy"), pred_rirs)

        if not args.skip_music:
            pred_music = evaluate.render_music(pred_rirs, D.music_dls)
            np.save(os.path.join(pred_dir,"pred_musics.npy"), pred_music)
    else:
        pred_rirs = np.load(os.path.join(pred_dir, "pred_rirs.npy"))
        pred_music = np.load(os.path.join(pred_dir, "pred_musics.npy"))



    """
    Evaluation of Monoaural Audio Using Metrics
    """
    if not args.skip_eval:
        errors_dir = os.path.join(args.save_dir, "errors")
        makedir_if_needed(errors_dir)
        list_of_metrics = metrics.baseline_metrics

        if args.valid:
            eval_indices = D.valid_indices
        else:
            eval_indices = D.test_indices

        # Evaluating RIR Interp
        for eval_metric in list_of_metrics:

            metric_name = eval_metric.__name__
            errors = evaluate.compute_error(pred_rirs, gt_audio, metric=eval_metric)
            np.save(os.path.join(errors_dir, "errors_" + metric_name +".npy"), errors)
            print(metric_name + " Metric:", flush=True)
            print(np.mean(errors[eval_indices]))
        
        # Evaluating Music Interp
        if not args.skip_music:
            for eval_metric in list_of_metrics:

                metric_name = eval_metric.__name__

                # Computing Error
                errors_music = evaluate.eval_music(pred_music, D.music, eval_metric)
                np.save(os.path.join(errors_dir, "errors_music_" + metric_name +".npy"), errors_music)
                print(metric_name + "Music Metric:", flush=True)
                print(np.mean(errors_music[eval_indices]))


    """
    Binaural Rendering
    """
    if not args.skip_binaural:
        pred_binaural_RIRs = []
        for i in range(D.bin_xyzs.shape[0]):
            binaural_RIR_xyz = D.bin_xyzs[i]
            bin_rir = binauralize.render_binaural(R=R, source_xyz = D.speaker_xyz,
                                                source_axis_1=None, source_axis_2=None,
                                                listener_xyz=binaural_RIR_xyz,
                                                listener_forward=D.default_binaural_listener_forward, 
                                                listener_left=D.default_binaural_listener_left,
                                                surfaces=D.all_surfaces,
                                                speed_of_sound=D.speed_of_sound,
                                                parallel_surface_pairs=D.parallel_surface_pairs,
                                                max_order=D.max_order, max_axial_order=D.max_axial_order)

            pred_binaural_RIRs.append(bin_rir)

        pred_binaural_RIRs = np.array(pred_binaural_RIRs)
        np.save(os.path.join(pred_dir, "pred_bin_RIRs.npy"), pred_binaural_RIRs)

        if not args.skip_music:
            pred_L = pred_binaural_RIRs[:,0,:]
            pred_R = pred_binaural_RIRs[:,1,:]

            pred_L_music = evaluate.render_music(pred_L, D.music_dls[:pred_L.shape[0]])
            pred_R_music = evaluate.render_music(pred_R, D.music_dls[:pred_R.shape[0]])

            pred_bin_music = np.stack((pred_L_music, pred_R_music), axis=2)
            print(pred_bin_music.shape)
            np.save(os.path.join(pred_dir, "pred_bin_musics.npy"), pred_bin_music)