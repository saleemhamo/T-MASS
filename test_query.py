import os
import gc
import torch
import random
import numpy as np
import pickle
from tqdm import tqdm
from transformers import CLIPTokenizer
from config.all_config import AllConfig
from datasets.msrvtt_dataset import MSRVTTDataset
from torch.utils.data import DataLoader
from datasets.model_transforms import init_transform_dict
from model.model_factory import ModelFactory
from trainer.trainer_stochastic import Trainer
from modules.metrics import sim_matrix_inference_stochastic_light_allops, generate_embeds_per_video_id_stochastic, \
    np_softmax
from config.all_config import gen_log

CACHE_DIR = "./cache"


def load_model(config):
    model = ModelFactory.get_model(config)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    if config.load_epoch is not None:
        checkpoint_path = os.path.join(config.model_path, f"checkpoint-epoch{config.load_epoch}.pth")
        if config.load_epoch == 0:
            checkpoint_path = os.path.join(config.model_path, "model_best.pth")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model, tokenizer


def process_query(query, tokenizer):
    inputs = tokenizer(query, return_tensors="pt")
    return inputs


def load_data(config):
    img_transforms = init_transform_dict(config.input_res)
    dataset = MSRVTTDataset(config, split_type='test', img_transforms=img_transforms['clip_test'])
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return data_loader


def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        print(f"Loaded cache from {cache_file}")
    else:
        cache = {}
    return cache


def save_cache(cache, cache_file):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Saved cache to {cache_file}")


def find_top_k_matches(query, model, tokenizer, data_loader, cache, k=5):
    text_inputs = process_query(query, tokenizer)

    with torch.no_grad():
        text_features = model.clip.get_text_features(
            input_ids=text_inputs['input_ids'].cuda(),
            attention_mask=text_inputs['attention_mask'].cuda()
        )

        all_text_embed_stochastic = []
        for trial in range(config.stochasic_trials):
            text_embed_stochastic, _, _ = model.stochastic(text_features, text_features)
            all_text_embed_stochastic.append(text_embed_stochastic)
        text_features = torch.stack(all_text_embed_stochastic, dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        vid_embeds_per_video_id = {}
        for batch in data_loader:
            video_ids = batch['video_id']
            video_features_list = []

            for idx, video_id in enumerate(video_ids):
                if video_id in cache:
                    video_features = cache[video_id].cuda()
                else:
                    video_data = batch['video'][idx].unsqueeze(0).cuda()
                    _, num_frames, channels, height, width = video_data.shape

                    if channels != 3:
                        raise ValueError(f"Expected 3 channels (RGB), but got {channels} channels.")

                    video_data = video_data.view(-1, channels, height, width)
                    video_features = model.clip.get_image_features(video_data)
                    cache[video_id] = video_features.cpu()

                video_features_list.append(video_features)

            video_features_tensor = torch.stack(video_features_list).squeeze()

            if video_features_tensor.dim() == 3:
                video_features_tensor = video_features_tensor.mean(dim=1)

            for i, video_id in enumerate(video_ids):
                vid_embeds_per_video_id[video_id] = video_features_tensor[i]

        model.pool_frames.cpu()
        vid_embeds_pooled = model.pool_frames(text_features[0], torch.stack(list(vid_embeds_per_video_id.values())))
        model.pool_frames.cuda()

        text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id_stochastic(
            text_features,
            vid_embeds_pooled,
            list(vid_embeds_per_video_id.keys()),
            config.pooling_type
        )

        sims = sim_matrix_inference_stochastic_light_allops(
            text_embeds_per_video_id,
            vid_embeds_pooled_per_video_id,
            config.pooling_type,
            config.batch_size_split,
            config
        )

        if config.DSL:
            sims = sims * np_softmax(sims * 100, axis=0)

        sorted_indices = torch.argsort(sims[0], descending=True)
        top_videos = [(list(vid_embeds_per_video_id.keys())[i], sims[0, i].item()) for i in sorted_indices[:k]]

    return top_videos


def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    writer = None

    if config.gpu is not None and config.gpu != '99':
        print('set GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model, tokenizer = load_model(config)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    data_loader = load_data(config)
    cache_file = os.path.join(CACHE_DIR, f"{config.dataset_name}_video_features.pkl")
    cache = load_cache(cache_file)

    top_videos = find_top_k_matches(config.query, model, tokenizer, data_loader, cache, k=5)
    print(f"Top 5 matching videos for the query '{config.query}':")
    for video_id, score in top_videos:
        print(f"Video ID: {video_id}, Score: {score}")

    save_cache(cache, cache_file)


if __name__ == '__main__':
    main()
