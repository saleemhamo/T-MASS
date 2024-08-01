import os
import torch
import argparse
from transformers import CLIPTokenizer
from model.model_factory import ModelFactory
from datasets.msrvtt_dataset import MSRVTTDataset
from torch.utils.data import DataLoader
from config.all_config import AllConfig
from datasets.model_transforms import init_transform_dict


def load_model(config):
    """Load the trained model and tokenizer."""
    model = ModelFactory.get_model(config)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    if config.load_epoch is not None:
        checkpoint_path = os.path.join(config.model_path, f"checkpoint-epoch{config.load_epoch}.pth")
        if config.load_epoch == 0:
            checkpoint_path = os.path.join(config.model_path, "model_best.pth")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model, tokenizer


def process_query(query, tokenizer):
    """Tokenize the text query."""
    inputs = tokenizer(query, return_tensors="pt")
    return inputs


def load_data(config):
    """Load and preprocess the video data from MSR-VTT dataset."""
    img_transforms = init_transform_dict(config.input_res)
    dataset = MSRVTTDataset(config, split_type='test', img_transforms=img_transforms['clip_test'])
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return data_loader


def find_best_match(query, model, tokenizer, data_loader):
    """Find the best matching video for the given query."""
    text_inputs = process_query(query, tokenizer)

    best_match_score = -float('inf')
    best_match_video = None

    with torch.no_grad():
        # Using the correct method to encode text
        text_features = model.clip.get_text_features(input_ids=text_inputs['input_ids'].cuda(),
                                                     attention_mask=text_inputs['attention_mask'].cuda())

        for batch in data_loader:
            video_features = batch['video'].cuda()
            batch_size, num_frames, channels, height, width = video_features.shape

            # Reshape video frames to the expected format: [batch_size * num_frames, channels, height, width]
            video_features = video_features.view(batch_size * num_frames, channels, height, width)

            # Ensure the channels are correct (3 for RGB images)
            if channels != 3:
                raise ValueError(f"Expected 3 channels (RGB), but got {channels} channels.")

            # Using the correct method to encode video
            video_features = model.clip.get_image_features(video_features)

            similarities = torch.matmul(text_features, video_features.t())

            max_score, max_index = similarities.max(dim=1)

            if max_score > best_match_score:
                best_match_score = max_score
                best_match_video = batch['video_id'][max_index].item()

    return best_match_video


def main():
    # Load configuration from AllConfig, which parses command-line arguments
    config = AllConfig()

    # Set the model path based on the parsed arguments
    config.model_path = os.path.join(config.output_dir, config.exp_name, config.datetime)

    # Set the device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model(config)
    model.to(device)

    # Load data
    data_loader = load_data(config)

    # Find the best matching video
    best_match_video = find_best_match(config.query, model, tokenizer, data_loader)
    print(f"The best matching video ID for the query '{config.query}' is: {best_match_video}")


if __name__ == '__main__':
    main()
