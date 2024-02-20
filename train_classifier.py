import torch 
import numpy as np 
from utils.data_utils import MR_US_dataset
from torch.utils.tensorboard import SummaryWriter
import os 
from utils.data_utils import *
from utils.train_utils import * 
from models.networks import * 
from utils.augmentation_utils import * 
import argparse 
CUDA_LAUNCH_BLOCKING = "1"

parser = argparse.ArgumentParser(prog='train',
                                description="Script for training classifier")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='efficient_mr',
                    help='Log dir to save results to')

parser.add_argument('--num_epochs',
                    metavar='num_epochs',
                    type=str,
                    default='10000',
                    action = 'store',
                    help='How many epochs to play the games for')

parser.add_argument('--data_type',
                    metavar='data_type',
                    type=str,
                    default='mr',
                    action = 'store',
                    help='Train with US or MR slices')

parser.add_argument('--batch_size',
                    metavar='batch_size',
                    type=str,
                    default='8',
                    action = 'store',
                    help='Minibatch size')

parser.add_argument('--pos_weight',
                    metavar='pos_weight',
                    type=str,
                    default='0.4',
                    action = 'store',
                    help='Pos weight applied for training')


parser.add_argument('--sample_fair',
                    metavar='sample_fair',
                    type=str,
                    default='True',
                    action = 'store',
                    help='Whether ot sapmle 0.5 of time with prostate and without prostaet')


parser.add_argument('--base_model',
                    metavar='base_model',
                    type=str,
                    default='resnet',
                    action = 'store',
                    help='Which basemodel to use : efficientnet or resnet 50')


parser.add_argument('--transform',
                    metavar='transform',
                    type=str,
                    default='default',
                    action = 'store',
                    help='Which transforms to use : jitter or no jitter')



# Parse arguments
args = parser.parse_args()

def save_images_2d(generated_imgs, save_folder, gt = None):
    """
    A function that saves generated images to folder 
    
    generated_img : in size batch_size x 10
    """
    
    os.makedirs(save_folder, exist_ok = True)
    
    # Obtain batch size : 
    batch_size = generated_imgs.size()[0]
    num_slices = generated_imgs 
    
    # Obtain final timestep (ie t = 100 to get denoised image)
    # Save 8 slices sas png files 
    num_slices = generated_imgs.size()[0]
    all_imgs = generated_imgs.squeeze()


    plt.figure()
    
    if gt != None:
        gt = gt.squeeze()
        
        fig, axs = plt.subplots(1,2)

        for slice in range(num_slices):
            #print(f"Slice: {slice}")
            #img = final_vol[:,:,:,slice]
            print(f'Shape of imgs : {all_imgs[:,:,slice].size()}')
            axs[0].imshow(all_imgs[slice,:,:].cpu().numpy())
            axs[0].axis('off')
            axs[1].imshow(gt[slice,:,:].cpu().numpy())
            img_name = 'slice_' + str(slice) + '.png'
            img_path = os.path.join(save_folder, img_name)
            plt.savefig(img_path)
            
    else:   
        for slice in range(num_slices):
            #print(f"Slice: {slice}")
            #img = final_vol[:,:,:,slice]
            plt.imshow(all_imgs[slice,:,:].cpu().numpy())
            plt.axis('off')
            img_name = 'slice_' + str(slice) + '.png'
            img_path = os.path.join(save_folder, img_name)
            plt.savefig(img_path)
        
        print(f"Saved img slice {slice}")
    
    plt.close() 
    

def compute_binary_accuracy(true_labels, predicted_labels, threshold = 0.4):
    """
    Compute accuracy for binary classification based on true and predicted labels.

    Parameters:
    - true_labels (list or array-like): True binary labels (1 or 0).
    - predicted_labels (list or array-like): Predicted binary labels (1 or 0).

    Returns:
    - positive_accuracy (float): Accuracy for positive (1) label pairs.
    - negative_accuracy (float): Accuracy for negative (0) label pairs.
    """

    true_labels_tensor = true_labels
    predicted_labels_tensor = 1.0*(predicted_labels>threshold).squeeze()

    # Count total pairs
    total_pairs = len(true_labels)

    # Count accurate positive and negative predictions using element-wise comparison
    positive_pairs = torch.sum((true_labels_tensor == predicted_labels_tensor) & (true_labels_tensor == 1))
    negative_pairs = torch.sum((true_labels_tensor == predicted_labels_tensor) & (true_labels_tensor == 0))

    # Calculate accuracy for positive and negative label pairs (ie accuracy on each posiitve and negative values)
    total_pos = torch.sum((true_labels_tensor == 1))
    total_neg = torch.sum((true_labels_tensor == 0))
    positive_accuracy = positive_pairs / total_pos if total_pos > 0 else torch.tensor(0).to(true_labels_tensor.device)
    negative_accuracy = negative_pairs / total_neg if total_neg > 0 else torch.tensor(0).to(true_labels_tensor.device)

    print(f"positive accuracy : {positive_accuracy} negative : {negative_accuracy}")
    return positive_accuracy.item(), negative_accuracy.item()

# # Example usage:
# true_labels = [1, 0, 1, 0, 1]
# predicted_labels = [1, 1, 0, 0, 1]

# pos_accuracy, neg_accuracy = compute_binary_accuracy(true_labels, predicted_labels)

# print(f"Positive Label Accuracy: {pos_accuracy * 100:.2f}%")
# print(f"Negative Label Accuracy: {neg_accuracy * 100:.2f}%")

if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    BASE_FOLDER = 'DOWNSTREAM_TASKS'
    SAVE_FOLDER = os.path.join(BASE_FOLDER, args.log_dir)
    
    BATCH_SIZE = int(args.batch_size)
    NUM_EPOCHS = int(args.num_epochs)
    EVAL_STEPS = 10 
    SAVE_STEPS = 100 
    BATCH_SIZE = 8 
    DATA_TYPE = args.data_type
    SAMPLE_FAIR = (args.sample_fair == 'True')
    TRANSFORM_TYPE = args.transform
    print(f"Save folder : {SAVE_FOLDER} using data type : {DATA_TYPE} sample fair : {SAMPLE_FAIR}")
    
    # PREVIOUSLY : data_folder = './Data' ; alligned = False; but this produces sagittal images!!! 
    
    # Load dataloaders, models 
    data_folder = './evaluate/REGNET'
    #data_folder = './Data'
    #data_folder = './RegData'
    
    if DATA_TYPE == 'us':
        # Aligned = false rotates US images to axial view [:,:,axial_slice]
        train_dataset = US_dataset(data_folder, mode = 'train', alligned = True, sample_fair = SAMPLE_FAIR)
        val_dataset = US_dataset(data_folder, mode = 'val', alligned = True, sample_fair = SAMPLE_FAIR)
    else:
        train_dataset = MR_dataset(data_folder, mode = 'train', alligned = True, sample_fair = SAMPLE_FAIR)
        val_dataset = MR_dataset(data_folder, mode = 'val', alligned = True, sample_fair = SAMPLE_FAIR)

    train_dataloader = DataLoader(train_dataset, batch_size= BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Define models for training 
    classifier_net = ClassifierNet(1, 1, base_model = args.base_model)
    classifier_net = classifier_net.to(device)
    
    # Define optimisers and loss functions 
    pos_weight = torch.ones(1)*float(args.pos_weight) # 75 positive, 25 negative  
    pos_weight = pos_weight.to(device)
    print(f"pos weight : {pos_weight}")
    
    if SAMPLE_FAIR: 
        print(f"Use unweighted loss fn")
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        print(f"Use weighted loss fn with pos weight {pos_weight}")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    opt = torch.optim.Adam(classifier_net.parameters())
    
    # Define tensorboard and saving files 
    os.makedirs(SAVE_FOLDER, exist_ok=True)  #, exists_ok = True)
    writer = SummaryWriter(os.path.join(SAVE_FOLDER, 'runs')) 

    # Save initial losss 
    best_loss = torch.tensor(1000000)
    
    for epoch in range(NUM_EPOCHS):
        
        print(f"\n Epoch : {epoch}")
        # Train model 
        classifier_net.train()
        
        # Saving lists 
        loss_train = [] 
        loss_vals = []
        train_labels = [] 
        train_preds = [] 
        val_labels = [] 
        val_preds = [] 
        
        for idx, (slice, whole_slice, label) in enumerate(train_dataloader):
            
            slice = slice.to(device)
            label = label.to(device)
            
            # Compute predictions 
            if TRANSFORM_TYPE == 'jitter':
                slice_augment = transforms_2d_jitter(slice)
            else:
                # Use default : gamma with random afifne 
                slice_augment = transforms_2d(slice)
            
            # Save images for reference
            #save_images_2d(slice_augment, SAVE_FOLDER, slice)
        
            pred_label = classifier_net(slice_augment.float())
            
            # Save images 
            
            # Compute loss 
            opt.zero_grad() # zero gradients
            #print(pred_label)
            #print(label)
            loss_val = loss_fn(pred_label, label.unsqueeze(1))
            loss_val.backward()
            loss_train.append(loss_val.item())
            
            # Adjust weights
            opt.step()
            
            # Add pred labels
            train_labels.append(label.detach())
            
            # Checks if there are 0 dim preds (ie only prediction left)
            no_shape_preds = pred_label.squeeze().detach().size() == torch.Size([])
            if no_shape_preds:
                train_preds.append(pred_label.squeeze(0).detach())
            else:
                train_preds.append(pred_label.squeeze().detach())
        
        # Compute across all predictions within epoch
        # Save results 
        with torch.no_grad():
            tp, tn = compute_binary_accuracy(torch.cat(train_labels, axis =0), torch.cat(train_preds, axis = 0))
            mean_loss = torch.mean(torch.tensor(loss_train))
            
        print(f"Epoch : {epoch} mean loss : {mean_loss} tp : {tp} tn: {tn}")
        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.add_scalar('TP/train', tp, epoch)
        writer.add_scalar('TN/train', tp, epoch)
        
        if (epoch % EVAL_STEPS):
            
            classifier_net.eval()
            with torch.no_grad():
                
                # Evaluate 
                for idx, (us, us_slice, us_label) in enumerate(val_dataloader):
                        
                    us = us.to(device)
                    us_label = us_label.to(device)
                    
                    preds = classifier_net(us.float())
                    loss_eval = loss_fn(preds, us_label.unsqueeze(1))
                    
                    # Add pred labels
                    val_labels.append(us_label.detach())
                    
                    # Checks if there are 0 dim preds (ie only prediction left)
                    no_shape_preds_val = preds.squeeze().detach().size() == torch.Size([])
                    if no_shape_preds_val:
                        val_preds.append(preds.squeeze(0).detach())
                    else:
                        val_preds.append(preds.squeeze().detach())
                
                    #val_preds.append(preds.squeeze().detach())
                    
                    # Save to val losses 
                    loss_vals.append(loss_eval)
                    

                # Save to summary writer
                mean_loss = torch.mean(torch.tensor(loss_val))
                
                tp_val, tn_val = compute_binary_accuracy(torch.cat(val_labels), torch.cat(val_preds))
                print(f"Epoch {epoch} : VALSAVEIDATION mean loss : {mean_loss} tp : {tp_val} tn : {tn_val}")
                writer.add_scalar('Loss/val', mean_loss, epoch)
                writer.add_scalar('TP/val', tp_val, epoch)
                writer.add_scalar('TN/val',tn_val, epoch)

                if mean_loss < best_loss:
                    print(f"Saving new model with loss : {mean_loss}")
                    val_path = os.path.join(SAVE_FOLDER, 'best_val_model.pth')
                    torch.save(classifier_net.state_dict(), val_path)
                    best_loss = mean_loss 

        if (epoch % SAVE_STEPS):
            print(f"Saving model periodically : epoch {epoch}")
            train_path = os.path.join(SAVE_FOLDER, 'train_model.pth')
            torch.save(classifier_net.state_dict(), train_path)
                             
    
    print('fuecoco')