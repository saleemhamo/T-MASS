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
    model.load_state_dict(torch.load(os.path.join(config.model_path, f"model_best.pth")))
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
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
        text_features = model.encode_text(text_inputs['input_ids'].cuda())

        for batch in data_loader:
            video_features = batch['video'].cuda()
            video_features = video_features.view(video_features.size(0), -1, 512)

            video_features = model.encode_video(video_features)

            similarities = torch.matmul(text_features, video_features.t())

            max_score, max_index = similarities.max(dim=1)

            if max_score > best_match_score:
                best_match_score = max_score
                best_match_video = batch['video_id'][max_index].item()

    return best_match_video


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Text-to-Video Retrieval Test")
    parser.add_argument('--datetime', required=True, help="Folder name under MSR-VTT-9k")
    parser.add_argument('--arch', default='clip_stochastic', help="Model architecture")
    parser.add_argument('--videos_dir', required=True, help="Directory containing videos")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--noclip_lr', type=float, default=3e-5, help="Learning rate for non-CLIP parameters")
    parser.add_argument('--transformer_dropout', type=float, default=0.3, help="Dropout rate for transformer")
    parser.add_argument('--dataset_name', default='MSRVTT', help="Dataset name")
    parser.add_argument('--msrvtt_train_file', default='9k', help="MSR-VTT training file")
    parser.add_argument('--stochasic_trials', type=int, default=20, help="Number of stochastic trials")
    parser.add_argument('--gpu', default='0', help="GPU device id")
    parser.add_argument('--load_epoch', type=int, default=0, help="Epoch to load for checkpointing")
    parser.add_argument('--exp_name', default='MSR-VTT-9k', help="Experiment name")
    parser.add_argument('--query', required=True, help="Text query for retrieval")
    args = parser.parse_args()

    # Load configuration
    config = AllConfig()
    config.model_path = os.path.join('checkpoints', args.exp_name, args.datetime)
    config.videos_dir = args.videos_dir
    config.batch_size = args.batch_size
    config.num_workers = 4
    config.input_res = 224
    config.arch = args.arch
    config.noclip_lr = args.noclip_lr
    config.transformer_dropout = args.transformer_dropout
    config.dataset_name = args.dataset_name
    config.msrvtt_train_file = args.msrvtt_train_file
    config.stochasic_trials = args.stochasic_trials
    config.gpu = args.gpu

    # Set the device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model(config)
    model.to(device)

    # Load data
    data_loader = load_data(config)

    # Find the best matching video
    best_match_video = find_best_match(args.query, model, tokenizer, data_loader)
    print(f"The best matching video ID for the query '{args.query}' is: {best_match_video}")


if __name__ == '__main__':
    main()
