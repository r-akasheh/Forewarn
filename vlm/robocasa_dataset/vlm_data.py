import h5py
import datasets
from datasets import SplitGenerator
from PIL import Image
import io
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import copy
import torch
from datasets import load_dataset

def numpy_to_jpeg_bytes(img):
    """
    Converts a NumPy array to JPEG bytes.
    """
    if img.dtype != 'uint8':
        img = img.astype('uint8')
    pil_img = Image.fromarray(img, 'RGB')
    with io.BytesIO() as buffer:
        pil_img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
    return img_bytes

def convert_images_to_bytes_parallel(images, max_workers=None):
    """
    Converts a list of NumPy images to JPEG bytes using parallel processing.
    
    Args:
        images (list or np.ndarray): List or array of NumPy images.
        max_workers (int, optional): Number of worker processes. Defaults to CPU count.
    
    Returns:
        list: List of JPEG bytes.
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pil_bytes = list(executor.map(numpy_to_jpeg_bytes, images))
    
    return pil_bytes
def numpy_to_pil_batch(batch):
    """
    Converts a batch of NumPy arrays to PIL Images.
    """
    pil_batch = []
    for img in batch:
        if img.dtype != 'uint8':
            img = img.astype('uint8')
        pil_batch.append(Image.fromarray(img, 'RGB'))
    return pil_batch

def convert_images_in_batches(images, batch_size=1000, max_workers=None):
    """
    Converts images in batches using parallel processing.
    
    Args:
        images (list or ndarray): List or array of NumPy images.
        batch_size (int): Number of images per batch.
        max_workers (int, optional): Number of worker processes. Defaults to CPU count.
    
    Returns:
        list: List of PIL Images.
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Split images into batches
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    
    pil_images = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for pil_batch in executor.map(numpy_to_pil_batch, batches):
            pil_images.extend(pil_batch)
    
    return pil_images
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for VQA generation task.

    Args:
        eval_pred: A tuple containing (predictions, labels)
            - predictions: Generated token IDs from the model
            - labels: True token IDs

    Returns:
        A dictionary with metric names as keys and metric values as values.
    """
    decoded_preds, decoded_labels= eval_pred


    # Convert to lowercase for case-insensitive comparison
    decoded_preds = [pred.lower() for pred in decoded_preds]
    decoded_labels = [label.lower() for label in decoded_labels]

    # Calculate accuracy: exact match
    accuracy = accuracy_score(decoded_labels, decoded_preds)

    # For binary metrics, map to binary labels
    label_mapping = {"failure": 0, "success": 1}
    binary_preds = [label_mapping.get(pred, -1) for pred in decoded_preds]
    binary_labels = [label_mapping.get(label, -1) for label in decoded_labels]

    # Remove instances where prediction is invalid (-1)
    valid_indices = [i for i, pred in enumerate(binary_preds) if pred != -1]
    binary_preds = [binary_preds[i] for i in valid_indices]
    binary_labels = [binary_labels[i] for i in valid_indices]

    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, average='binary', pos_label=1
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
def check_header(targets,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False
def replace_target(target,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i],seq[i+1],seq[i+2] = -100,-100,-100
    return seq
def process(examples, processor):
    # text_begin = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    # text_end = "{example['question']} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['answer']}<|eot_id|>"
    dialog = []
    images = []
    for example in examples:
        length = len(example['images'])
        # breakpoint()
        current_dialog = [ {"role":"user","content":[{"type": "image"}]},
                    {"role":"assistant","content":[{"type": "text", "text": example['answer'].strip()}]}]
        ## for each image, add an image token
        # for i in range(length-1):
        #     current_dialog[0]["content"].append({"type": "image"})
        current_dialog[0]["content"].append({"type": "text", "text": example['question'].strip()})
        dialog.append(current_dialog)
        images.append(example["images"])
    # breakpoint()
    batch = tokenize_dialogs(dialog, images, processor)

    return batch
def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list, device='cuda:0') #torch.tensor(label_list)
    return batch

class HDF5VQADataset(datasets.GeneratorBasedBuilder):
    """Custom Dataset for VQA-like tasks from HDF5 files."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="A VQA-like dataset constructed from HDF5 files containing robot trajectories.",
            features=datasets.Features({
                'images_bytes': datasets.Sequence(datasets.Value("binary")),#datasets.Sequence(datasets.Image()),  # Sequence of images
                'question': datasets.Value('string'),
                "answer": datasets.Value('string'),
                'images': datasets.Sequence(datasets.Image())
            }),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        # Define the splits based on provided HDF5 files
        # Expecting data_files to have 'train', 'validation', 'test' keys
        data_files = self.config.data_files
        if 'train' in data_files:
            return [
                SplitGenerator(
                    name="train",
                    gen_kwargs={"hdf5_paths": data_files["train"]}
                )]
        else:
            # SplitGenerator(
            #     name="validation",
            #     gen_kwargs={"hdf5_path": data_files["validation"]}
            # ),
            return[SplitGenerator(
                name="test",
                gen_kwargs={"hdf5_paths": data_files["test"]}
            ),
        ]

    def _generate_examples(self, hdf5_paths):
        """Yields examples from the HDF5 file."""
        for hdf5_path in hdf5_paths:
            init_time = time.time()
            with h5py.File(hdf5_path, 'r') as f:
                trajectories = f['data'].keys()  # Assuming each trajectory is a key
                # data = f['data']
                for traj_id, traj in enumerate(list(trajectories)[:4]):
                    ## two views of the same trajectory
                    wrist_images = f['data'][traj]['obs']['robot0_eye_in_hand_image'][-1:]
                    front_images = f['data'][traj]['obs']['robot0_agentview_front_image'][-1:]
                    ## concatenate the two views
                    # start_time = time.time()
                    # images = wrist_images
                    images = np.concatenate((wrist_images, front_images), axis=1)
                    # print('Time taken for concatenation :', time.time() - start_time)
                    # images = f[traj]['images'][:]  # Shape: (1500, H, W, C)
                    # actions = f[traj]['actions'][:]  # Not used in VQA
                    # robot_states = f[traj]['robot_states'][:]  # Not used in VQA
                    label = f['data'][traj].attrs['label']  # 0 or 1

                    # Convert label to 'A' or 'B'
                    answer = 'Failure' if label == 0 else 'Success'

                    # Convert numpy images to PIL Images
                    # start_time = time.time()
                    # pil_images = [Image.fromarray(img.astype('uint8'), 'RGB') for img in images]
                    pil_images = convert_images_to_bytes_parallel(images)
                    # pil_images = Image.open('/home/yilin/Projects/failure_detection/vlm/success_exp_traj_100.gif')
                    # pil_images = convert_images_in_batches(images)
                    # print('Time taken for pil :', time.time() - start_time)
                    question = "The robot is trying to put the cup from the counter and place it into the sink. You are a runtime montior trying to monitor the task progress. The images is a sequences of observations for the robot execution for this task. Each image frame is composed of wrist view (top) and front view (bottom) of the robot. From the image sequences, summarize if the task is success or failure. Answer is one word either failure or success."
                    print('Time taken for one traj :', time.time() - init_time)
                    # start_time = time.time()    
                    yield f"{hdf5_path}-{traj_id}", {
                        'images_bytes': pil_images,#f['data'][traj]['obs']['robot0_eye_in_hand_image'],#images,
                        'question': question,
                        'answer': answer,
                    }
                    # print('Time taken for yield :', time.time() - start_time)   
def decode_images(example):
    """
    Decodes JPEG bytes back into PIL Images.
    
    Args:
        example (dict): A single example from the dataset.
    
    Returns:
        dict: Example with decoded images.
    """
    decoded_images = []
    for img_bytes in example['images_bytes']:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        decoded_images.append(pil_img)
    # example['images'] = decoded_images
    new_dict = {}
    new_dict['images'] = decoded_images
    print('Decoded images', new_dict['images'][0])
    # print('not decoded images', example['images_bytes'][0])
    # del example['images_bytes']  # Remove bytes to save memory
    return new_dict

def get_custom_dataset(train, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    if train:
        dataset = load_dataset(
            'robocasa_dataset/vlm_data.py',
            data_files={
                'train': 'train.hdf5',
                # 'validation': 'validation.h5',
                # 'test': 'eval.hdf5'
            },
            trust_remote_code=True
        )
        dataset = dataset.map(decode_images, batched=False, remove_columns=["images_bytes"], num_proc=10)
        return dataset['train']
    else:
        dataset = load_dataset(
            'robocasa_dataset/vlm_data.py',
            data_files={
                # 'train': 'train.hdf5',
                # 'validation': 'validation.h5',
                'test': 'eval.hdf5'
            },
            trust_remote_code=True
        )
        dataset = dataset.map(decode_images, batched=False, remove_columns=["images_bytes"], num_proc=10)
        return dataset['test']
    
def get_data_collator(processor):
    return lambda x: process(x, processor)