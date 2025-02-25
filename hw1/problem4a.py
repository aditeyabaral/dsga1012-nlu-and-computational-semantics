import json
import argparse
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from embeddings import Embeddings
from test_analogies import (
    get_closest_words,
    load_analogies,
    AnalogiesDataset,
)


def load_embeddings() -> Dict[int, Embeddings]:
    """
    Loads GloVe embeddings of dimensions 50, 100, and 200 from the
    data/ directory. Returns a dictionary that maps each embedding
    dimension to the corresponding Embeddings object."""
    embeddings = {
        dim: Embeddings.from_file(f"data/glove_{dim}d.txt") for dim in [50, 100, 200]
    }
    return embeddings


def load_analogies_subsets(
    filename: str = "data/analogies.txt",
) -> Dict[str, AnalogiesDataset]:
    """
    Loads testing data for 3CosAdd from a .txt file and deserializes it
    into the AnalogiesData format. Returns the full dataset as well as
    the semantic and syntactic subsets of the dataset.
    """
    test_data = load_analogies(filename)

    semantic_test_data = {
        k: test_data[k]
        for k in [
            "capital-common-countries",
            "capital-world",
            "currency",
            "city-in-state",
            "family",
        ]
    }

    syntactic_test_data = {
        k: test_data[k]
        for k in [
            "gram1-adjective-to-adverb",
            "gram2-opposite",
            "gram3-comparative",
            "gram4-superlative",
            "gram5-present-participle",
            "gram6-nationality-adjective",
            "gram7-past-tense",
            "gram8-plural",
            "gram9-plural-verbs",
        ]
    }
    return {
        "overall": test_data,
        "semantic": semantic_test_data,
        "syntactic": syntactic_test_data,
    }


def run_analogy_test(
    embeddings: Embeddings,
    analogies: List[Tuple[str, str, str, str]],
    verbose: str,
    k: int = 1,
) -> Tuple[int, int, float]:
    """
    Runs the 3CosAdd test for a set of embeddings and analogy questions.

    :param embeddings: The embeddings to be evaluated using 3CosAdd
    :param analogies: A list of analogy questions, where each
        analogy is represented as a tuple of four strings
    :param k: The "lenience" with which accuracy will be computed. An
        analogy question will be considered to have been answered
        correctly as long as the target word is within the top k closest
        words to the target vector, even if it is not the closest word
    :return: The results of the 3CosAdd test, in the format of a tuple
        containing the number of correctly answered questions, the total
        number of questions, and the accuracy of the model
    """
    correct = 0
    total = 0
    for analogy in tqdm(analogies, desc=verbose):
        w1, w2, w3, w4 = analogy
        vectors = embeddings[w1, w2, w3]
        relation = [vectors[1] - vectors[0] + vectors[2]]
        closest_words = get_closest_words(embeddings, relation, k)[0]
        if w4 in closest_words:
            correct += 1
        total += 1
    accuracy = correct / total
    return correct, total, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Problem 4a")
    parser.add_argument("--lenience", "-k", type=int, default=1)
    args = parser.parse_args()

    results = dict()
    embeddings = load_embeddings()
    analogies_subsets = load_analogies_subsets()
    for dim, embedding in embeddings.items():
        # print(f"Results for {dim}-dimensional embeddings:")
        results[dim] = dict()
        for subset_type, subset_data in analogies_subsets.items():
            # print(f"Results for {subset_type} analogies:")
            concatenated_analogies = [
                analogy for analogies in subset_data.values() for analogy in analogies
            ]
            correct, total, accuracy = run_analogy_test(
                embedding,
                concatenated_analogies,
                verbose=f"{dim}d / {subset_type} analogies",
                k=args.lenience,
            )
            # print(f"Accuracy: {accuracy:.3f}")
            results[dim][subset_type] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
            print()
        print()

    print("Results summary:")
    for dim, dim_results in results.items():
        print(f"{dim}-dimensional embeddings:")
        for subset_type, subset_results in dim_results.items():
            print(f"{subset_type} analogies:")
            print(
                f"Accuracy: {subset_results['accuracy']:.3f} "
                f"({subset_results['correct']} / {subset_results['total']})"
            )
        print()

    with open(f"results4a_k{args.lenience}.json", "w") as f:
        json.dump(results, f, indent=4)
