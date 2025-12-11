# util/augmentation_second_best_mutation.py

import random
import numpy as np
from util.seed import set_seed
from util.blosum62_data import BLOSUM62_MATRIX_DATA, BLOSUM_AMINO_ACIDS_ORDER

set_seed()

# --- Original Insertion Function (random) ---
def insertion_sequence(seq_list, insertion_rate, amino_acids):
    num_insertions = np.random.binomial(len(seq_list), insertion_rate)
    for _ in range(num_insertions):
        insert_idx = np.random.randint(0, len(seq_list) + 1)
        seq_list.insert(insert_idx, random.choice(amino_acids))
    return seq_list

# --- Original Deletion Function (random) ---
def deletion_sequence(seq_list, deletion_rate, amino_acids): # Added amino_acids to match the call
    num_deletions = np.random.binomial(len(seq_list), deletion_rate)
    for _ in range(num_deletions):
        if len(seq_list) > 1:
            delete_idx = np.random.randint(0, len(seq_list))
            del seq_list[delete_idx]
    return seq_list


# --- NEW Mutation Function: Replaces with the HIGHEST-scoring DIFFERENT amino acid ---
def mutate_sequence_second_best(seq_list, mutation_rate, blosum_matrix, amino_acids_order):
    mutated_seq_list = list(seq_list)
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids_order)} 

    mutation_mask = np.random.rand(len(seq_list)) < mutation_rate 

    for local_idx, (mutate, original_aa) in enumerate(zip(mutation_mask, seq_list)):
        if mutate:
            original_aa_idx = aa_to_idx.get(original_aa)
            if original_aa_idx is None: 
                mutated_seq_list[local_idx] = original_aa
                continue

            scores_for_original_aa = blosum_matrix[original_aa_idx]

            # --- Core Logic: Find the highest-scoring DIFFERENT amino acid ---
            # 1. Create a temporary copy of scores, ensuring it is of FLOAT type to handle -np.inf
            temp_scores = np.copy(scores_for_original_aa).astype(float) # <-- SOLVES OverflowError
            
            # 2. Set the score of the original amino acid to negative infinity
            # This ensures that np.argmax will find the highest score among all OTHER amino acids.
            temp_scores[original_aa_idx] = -np.inf 

            # 3. Find the index of the best-scoring *different* amino acid
            best_different_idx = np.argmax(temp_scores) # <-- This now finds the highest score excluding the original AA
            
            # Get the amino acid character corresponding to this index
            new_aa = amino_acids_order[best_different_idx]
            
            mutated_seq_list[local_idx] = new_aa
        else:
            mutated_seq_list[local_idx] = original_aa
    return mutated_seq_list


# --- Main Augmentation Function ---
def augment_sequence_with_second_best_mutation(seq, num_fragments=6, mutation_rate=0.5, insertion_rate=0.5, deletion_rate=0.5, multi_step=1):
    seq_list = list(seq)

    fragment_points = sorted(random.sample(range(1, len(seq_list)), min(num_fragments - 1, len(seq_list) - 1)))
    fragments = [seq_list[i:j] for i, j in zip([0] + fragment_points, fragment_points + [None])]

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY") 

    for _ in range(multi_step):
        for i in range(len(fragments)):
            fragment = fragments[i]
            if i % 3 == 0:
                fragments[i] = mutate_sequence_second_best(fragment, mutation_rate, BLOSUM62_MATRIX_DATA, BLOSUM_AMINO_ACIDS_ORDER)
            elif i % 3 == 1: 
                fragments[i] = insertion_sequence(fragment, insertion_rate, amino_acids)
            else: 
                fragments[i] = deletion_sequence(fragment, deletion_rate, amino_acids)

    return ''.join(sum(fragments, []))