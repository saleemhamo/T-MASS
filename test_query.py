import os
import torch
import pandas as pd
import pickle
from transformers import CLIPTokenizer
from model.model_factory import ModelFactory
from datasets.msrvtt_dataset import MSRVTTDataset
from torch.utils.data import DataLoader
from config.all_config import AllConfig
from datasets.model_transforms import init_transform_dict
from stochastic_text_wrapper import StochasticTextWrapper

# Setup logging
import logging

logging.basicConfig(filename='evaluation.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

CACHE_FILE = 'video_features_cache.pkl'


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
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore missing keys
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Wrap the stochastic text module with the new wrapper
    model.stochastic = StochasticTextWrapper(config)

    return model, tokenizer


def process_query(query, tokenizer):
    """Tokenize the text query."""
    inputs = tokenizer(query, return_tensors="pt").to('cuda')
    return inputs


def load_data(config):
    """Load and preprocess the video data from MSR-VTT dataset."""
    img_transforms = init_transform_dict(config.input_res)
    dataset = MSRVTTDataset(config, split_type='test', img_transforms=img_transforms['clip_test'])
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return data_loader


def save_cache(cache, file_path):
    """Save the cache to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(cache, f)


def load_cache(file_path):
    """Load the cache from a file if it exists."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}


def find_top_k_matches(config, query, model, tokenizer, data_loader, video_features_cache, k=10):
    """Find the top-k matching videos for the given query."""
    text_inputs = process_query(query, tokenizer)
    text_features = model.clip.get_text_features(
        input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask']
    )

    video_scores = {}

    with torch.no_grad():
        for batch in data_loader:
            video_ids = batch['video_id']
            video_features = batch['video'].to('cuda')

            if video_features.dim() == 5:
                batch_size, num_frames, channels, height, width = video_features.shape
                video_features = video_features.view(batch_size * num_frames, channels, height, width)
            elif video_features.dim() == 4:
                batch_size, channels, height, width = video_features.shape
                num_frames = 1
            else:
                raise ValueError(f"Unexpected video features shape: {video_features.shape}")

            video_features = model.clip.get_image_features(video_features)

            if num_frames > 1:
                video_features = video_features.view(batch_size, num_frames, -1)

            for idx, video_id in enumerate(video_ids):
                if video_id not in video_features_cache:
                    video_data = video_features[idx].unsqueeze(0)
                    video_features_cache[video_id] = video_data.cpu()
                else:
                    video_data = video_features_cache[video_id].to('cuda')

                for trial in range(config.stochasic_trials):
                    aligned_text_features, _, _ = model.stochastic(text_features, video_data)

                    similarities = torch.matmul(aligned_text_features, video_data.mean(dim=1).t())
                    available_k = min(k, similarities.shape[1])
                    top_scores, top_indices = similarities.topk(available_k, dim=1)

                    for score, idx in zip(top_scores.cpu().numpy().flatten(), top_indices.cpu().numpy().flatten()):
                        video_id = video_ids[idx]
                        if video_id in video_scores:
                            video_scores[video_id] = max(video_scores[video_id], score)
                        else:
                            video_scores[video_id] = score

    # Sort video scores and get the top-k
    sorted_videos = sorted(video_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_videos[:k]


def evaluate_model_on_test_data(config, model, tokenizer, data_loader, test_data, k=10, limit=None):
    """Evaluate the model on test data."""
    video_features_cache = load_cache(CACHE_FILE)

    correct_at_k = [0] * k
    ranks = []
    total_queries = len(test_data) if limit is None else limit

    for i, (_, row) in enumerate(test_data.iterrows()):
        if limit is not None and i >= limit:
            break
        query = row['sentence']
        correct_video_id = row['video_id']
        top_videos = find_top_k_matches(config, query, model, tokenizer, data_loader, video_features_cache, k)
        top_video_ids = [video_id for video_id, _ in top_videos]

        for rank, video_id in enumerate(top_video_ids):
            if video_id == correct_video_id:
                ranks.append(rank + 1)
                for j in range(rank, k):
                    correct_at_k[j] += 1
                break
        else:
            ranks.append(k + 1)

        # Log progress
        logger.info(f"Processed query {i + 1}/{total_queries}: {query}")
        print(f"Processed query {i + 1}/{total_queries}: {query}")

    recall_at_k = [correct / total_queries for correct in correct_at_k]
    median_rank = torch.median(torch.tensor(ranks, dtype=torch.float)).item()
    mean_rank = torch.mean(torch.tensor(ranks, dtype=torch.float)).item()

    results = {
        "R@1": recall_at_k[0],
        "R@5": recall_at_k[4],
        "R@10": recall_at_k[9],
        "MdR": median_rank,
        "MnR": mean_rank
    }

    for metric, value in results.items():
        logger.info(f"{metric}: {value}")
        print(f"{metric}: {value}")

    save_cache(video_features_cache, CACHE_FILE)
    return results


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

    # Load test data
    test_data = pd.read_csv('data/MSRVTT/MSRVTT_JSFUSION_test.csv', names=['key', 'vid_key', 'video_id', 'sentence'])

    # Evaluate model on test data with a limit of 2 records for testing
    evaluate_model_on_test_data(config, model, tokenizer, data_loader, test_data, k=10)


if __name__ == '__main__':
    main()
