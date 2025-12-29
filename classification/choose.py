import re
from typing import List, Tuple

def parse_log_file(file_path: str) -> List[List[Tuple[float, float, float, str]]]:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expressions to extract required information
    genotype_pattern = re.compile(r"genotype = (.+)")
    validation_loss_pattern = re.compile(r"validation loss: ([\d\.]+)")
    prec1_pattern = re.compile(r"prec1: tensor\(\[([\d\.]+)\]")
    prec5_pattern = re.compile(r"prec5: tensor\(\[([\d\.]+)\]")

    all_generations = []
    current_generation = []

    for i in range(0, len(lines), 2):  # Assuming every two lines are related
        genotype_match = genotype_pattern.search(lines[i])
        val_loss_match = validation_loss_pattern.search(lines[i+1])
        prec1_match = prec1_pattern.search(lines[i+1])
        prec5_match = prec5_pattern.search(lines[i+1])

        if genotype_match and val_loss_match and prec1_match and prec5_match:
            genotype = genotype_match.group(1).strip()
            val_loss = float(val_loss_match.group(1))
            prec1 = float(prec1_match.group(1))
            prec5 = float(prec5_match.group(1))
            current_generation.append((val_loss, prec1, prec5, genotype))

        # Assuming each generation consists of 50 entries:
        if len(current_generation) == 50:
            all_generations.append(current_generation)
            current_generation = []

    return all_generations

def select_top_individuals(generation_data: List[Tuple[float, float, float, str]], num_individuals: int = 10) -> List[Tuple[float, float, float, str]]:
    # Sort by validation loss, then by prec1 and prec5 in descending order
    sorted_data = sorted(generation_data, key=lambda x: (x[0], -x[1], -x[2]))
    return sorted_data[:num_individuals]

def main():
    log_file_path = './kde_search10-20241227-221431/log.txt'
    all_generations = parse_log_file(log_file_path)

    for gen_index, generation_data in enumerate(all_generations):
        top_individuals = select_top_individuals(generation_data)

        print(f"Generation {gen_index + 1}:")
        for val_loss, prec1, prec5, genotype in top_individuals:
            print(f"Validation Loss: {val_loss}, Prec1: {prec1}, Prec5: {prec5}, Genotype: {genotype}")
        print()

if __name__ == "__main__":
    main()
