import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description='Parallel LLaVA evaluation script.')

    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file-gt", type=str, default="answer_gt.csv")
    parser.add_argument("--answers-file-pred", type=str, default="answer_pred.csv")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument('--num-chunks', type=int, default=1, help='Number of chunks (default: 1).')
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    return args

def run_job(chunk_idx, args):
    cmd = ("CUDA_VISIBLE_DEVICES={chunk_idx} python llava/eval/model_vqa_mimic_single.py "
           "--model-name {model_name} "
           "--question-file {question_file} "
           "--image-folder {image_folder} "
           "--answers-file-gt {experiment_name_with_split}-gt-chunk{chunk_idx}.csv "
           "--answers-file-pred {experiment_name_with_split}-pred-chunk{chunk_idx}.csv "
           "--num-chunks {chunks} "
           "--chunk-idx {chunk_idx} ").format(
                chunk_idx=chunk_idx,
                chunks=args.num_chunks,
                model_name=args.model_name,
                question_file=args.question_file,
                image_folder=args.image_folder,
                experiment_name_with_split=args.experiment_name_with_split
            )

    print(cmd)

    subprocess.run(cmd, shell=True, check=True)

def main():
    args = parse_args()
    args.experiment_name_with_split = args.answers_file_gt.split(".csv")[0]

    # Create a partial function that accepts only `chunk_idx`
    from functools import partial
    run_job_with_args = partial(run_job, args=args)

    # Run the jobs in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.num_chunks) as executor:
        list(executor.map(run_job_with_args, range(args.num_chunks)))  # Use run_job_with_args instead of lambda

    # Gather the results
    output_file_gt = f"{args.experiment_name_with_split}_gt.csv"
    output_file_pred = f"{args.experiment_name_with_split}_pred.csv"

    with open(output_file_gt, 'w') as outfile_gt, open(output_file_pred, 'w') as outfile_pred:
        for idx in range(args.num_chunks):
            with open(f"{args.experiment_name_with_split}-gt-chunk{idx}.csv") as infile_gt, open(f"{args.experiment_name_with_split}-pred-chunk{idx}.csv") as infile_pred:
                outfile_gt.write(infile_gt.read())
                outfile_pred.write(infile_pred.read())

if __name__ == "__main__":
    main()