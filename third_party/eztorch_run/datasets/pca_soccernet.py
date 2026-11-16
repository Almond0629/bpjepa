import argparse

from eztorch.datasets.soccernet import soccernet_dataset
from eztorch.datasets.soccernet_utils.features import sampled_pca_features_to_zip

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform PCA on SoccerNet features.")
    parser.add_argument("--dataset-json", type=str, help="Location dataset.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["action", "ball", "all"],
        help="Task of the dataset.",
    )
    parser.add_argument(
        "--video-zip-prefix", type=str, default="", help="Prefix of features in zip."
    )
    parser.add_argument(
        "--features-path",
        type=str,
        help="Path to the input features.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to output the PCA features.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Name of the features files.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="Dimension of the PCA ouput.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="FPS of the stored features.",
    )
    parser.add_argument(
        "--sample-per-half",
        type=int,
        default=256,
        help="Number of uniformly sampled rows per half used to fit PCA.",
    )
    parser.add_argument(
        "--transform-batch-size",
        type=int,
        default=512,
        help="Number of rows transformed at once.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for PCA sampling and randomized SVD.",
    )

    args = parser.parse_args()

    dataset = soccernet_dataset(
        data_path=args.dataset_json,
        transform=None,
        video_path_prefix="",
        decoder="frame",
        decoder_args={"fps": args.fps},
        label_args=None,
        features_args=None,
        task=args.task,
    )

    video_paths = [
        video_path.decode() for video_path in dataset._annotated_videos._video_paths
    ]

    output_zip = sampled_pca_features_to_zip(
        features_dir=args.features_path,
        video_paths=video_paths,
        filename=args.filename,
        saving_path=args.save_path,
        dim=args.dim,
        sample_per_half=args.sample_per_half,
        transform_batch_size=args.transform_batch_size,
        seed=args.seed,
        video_zip_prefix=args.video_zip_prefix,
    )
    print(f"saved {output_zip}")
