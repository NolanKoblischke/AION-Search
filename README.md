# AION-Search

### Note: Improved documentation will be available for the official release in later December.

This repository contains the implementation of "Why wait for human annotations when you have AI? Semantic searching scientific images with synthetic labels". The codebase demonstrates how Vision-Language Models can generate descriptions of galaxy images that serve as training data for a semantic search engine, enabling natural language queries over 140 million astronomical images without human annotation.

## Paper Implementation Overview

The paper presents three main contributions, each corresponding to different parts of the codebase. First, we evaluate whether VLMs can accurately describe galaxy images by benchmarking them against human annotations from Galaxy Zoo. Second, we use these VLM-generated descriptions to train a contrastive model that aligns image embeddings with text embeddings. Third, we demonstrate that VLM re-ranking can significantly improve retrieval performance for rare astronomical phenomena.

## VLM Benchmarking

Figure 1 shows the accuracy-cost trade-off for 13 different Vision-Language Models on our Galaxy Zoo benchmark. This analysis is implemented in `prompt_optimization/plot_performance_vs_cost.py`, which reads evaluation results from JSONL files containing model responses and their associated costs. The script loads pricing information from `src/utils/models.jsonl` and calculates the cost to caption 100,000 images for each model.

The benchmark itself evaluates VLMs on 64 carefully curated galaxy images from Galaxy Zoo DECaLS. The curation process, implemented in `galaxybench/dataprep/galaxyzoo/prep.py`, selects images with strong consensus among human annotators and ensures diversity across morphological types. The evaluation pipeline consists of three scripts in `galaxybench/eval/combined/`: first, `generate_descriptions.py` prompts VLMs to describe each galaxy image using the optimized prompt from `prompt_optimization/prompts/general_promptv4.txt`. Then `judge.py` uses Gemini-2.5-Flash to extract Galaxy Zoo decision tree answers from these free-form descriptions. Finally, `print_score.py` compares these extracted answers against the human consensus to compute accuracy scores.

## Training Pipeline

The core contribution of our paper is the AION-Search model, which enables semantic search over astronomical images. The training pipeline, implemented as a sequence of numbered scripts in `src/`, transforms unlabeled galaxy images into a searchable semantic space through six stages.

The pipeline begins with `01_collect_galaxies.py`, which samples 300,000 galaxy images from the Multi Modal Universe dataset, split between HSC and Legacy Survey telescopes. These images are selected using only a simple brightness cut to avoid biasing the model toward specific morphological types. Next, `02a_generate_descriptions_batch.py` uses OpenAI's batch API to generate descriptions for each image using GPT-4.1-mini with our optimized prompt. This script manages the asynchronous batch processing, handling retries and failures gracefully. The total cost for generating descriptions for 300,000 images is approximately $150.

Following description generation, `03a_generate_augmented_descriptions_batch.py` creates single-sentence summaries of the longer descriptions using GPT-4.1-nano. The script `04_generate_text_embeddings.py` then embeds both the original descriptions and summaries using OpenAI's text-embedding-3-large model, producing 3072-dimensional vectors that capture semantic content.

The preparation phase concludes with `05_generate_unified_embeddings.py`, which combines the text embeddings with pre-computed AION image embeddings. This script handles the alignment of different data sources, filters out images used in downstream benchmarks, and creates the final training dataset. Finally, `06_train_clip.py` implements the contrastive learning that aligns AION's image encoder with the text embedding space. The training uses shallow MLPs as projection heads and optimizes an InfoNCE loss to learn a shared 1024-dimensional embedding space.

## Evaluation Infrastructure

The paper's main results comparing AION-Search to baselines appear in Figure 2 and the accompanying table. These evaluations are implemented in `src/experiments/aion_table4/`, which contains separate scripts for different target phenomena. The evaluation follows AION's established protocol, using nDCG@10 to measure retrieval quality on datasets with varying rarity: spiral galaxies (26% of dataset), mergers (2%), and gravitational lenses (0.1%).

For spirals and mergers, the evaluation uses Galaxy Zoo DECaLS labels where each image has a relevance score equal to the fraction of volunteers identifying that feature. The script `eval_table4_gz.py` loads the trained AION-Search model, generates embeddings for text queries like "visible spiral arms" and "merging", then computes retrieval metrics against the labeled dataset.

Gravitational lens evaluation, implemented in `eval_table4_lens.py`, presents a more challenging test case due to their extreme rarity. The evaluation dataset combines confirmed lenses from published catalogs with a large set of non-lenses, assigning binary relevance scores. The dramatic improvement of AION-Search over similarity-based methods on this task demonstrates the value of semantic search for finding rare phenomena.

## Re-ranking Experiments

Figure 3 presents our re-ranking results, showing how VLMs can verify and improve initial search results. The implementation in `src/experiments/rerank/` explores two dimensions of scaling: model capacity (from GPT-4.1-nano through GPT-4.1) and inference-time compute through average sampling.

The main experiment script, `rerank_multi_experiments.py`, first uses AION-Search to retrieve the top 1000 candidates for gravitational lenses. It then prompts different VLMs to score each candidate on a 1-10 scale based on whether they show signs of lensing. For the sampling experiments, the script generates multiple independent scores per image and averages them. The results demonstrate that both larger models and increased sampling improve performance, nearly doubling the number of confirmed lenses in the top 100 results.

The visualization script `plot_rerank_vs_baseline.py` creates the paper's Figure 3, showing both the performance scaling and examples of top-ranked lenses with excerpts from the VLM explanations.