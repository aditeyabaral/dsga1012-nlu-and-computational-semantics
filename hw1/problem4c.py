import json
import argparse
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from embeddings import Embeddings
from test_analogies import get_closest_words


def load_embeddings() -> Dict[int, Embeddings]:
    """
    Loads GloVe embeddings of dimensions 50, 100, and 200 from the
    data/ directory. Returns a dictionary that maps each embedding
    dimension to the corresponding Embeddings object."""
    embeddings = {
        dim: Embeddings.from_file(f"data/glove_{dim}d.txt") for dim in [50, 100, 200]
    }
    return embeddings


def run_analogy_test(
    embeddings: Embeddings,
    analogies: List[Tuple[str, str, str, str]],
    k: int = 1,
) -> List[str]:
    """
    Runs the 3CosAdd test for a set of embeddings and analogy questions.

    :param embeddings: The embeddings to be evaluated using 3CosAdd
    :param analogies: A list of analogy questions, where each
        analogy is represented as a tuple of four strings
    :param k: The "lenience" with which accuracy will be computed. An
        analogy question will be considered to have been answered
        correctly as long as the target word is within the top k closest
        words to the target vector, even if it is not the closest word
    :return: The results of the 3CosAdd test, in the format of a list
        containing the predicted word for each analogy
    """
    results = []
    for analogy in analogies:
        w1, w2, w3, w4 = analogy
        vectors = embeddings[w1, w2, w3]
        relation = [vectors[1] - vectors[0] + vectors[2]]
        closest_words = get_closest_words(embeddings, relation, k)[0]
        results.append(closest_words)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Problem 4c")
    parser.add_argument("--lenience", "-k", type=int, default=1)
    args = parser.parse_args()

    results = dict()
    embeddings = load_embeddings()

    analogies = [
        ("france", "paris", "italy", "rome"),
        ("france", "paris", "japan", "tokyo"),
        ("france", "paris", "florida", "miami"),
        ("big", "bigger", "small", "smaller"),
        ("big", "bigger", "cold", "colder"),
        ("big", "bigger", "quick", "quicker"),
    ]

    for dim, embedding in embeddings.items():
        results[dim] = dict()
        analogy_results = run_analogy_test(embedding, analogies, k=args.lenience)
        results[dim] = {
            f"{k[0]} : {k[1]} :: {k[2]} :": v
            for k, v in zip(analogies, analogy_results)
        }

    for dim, result in results.items():
        print(f"Results for {dim}-dimensional embeddings:")
        for analogy, prediction in result.items():
            print(f"{analogy} {prediction[0]}")
        print()

    with open(f"results4c_k{args.lenience}.json", "w") as f:
        json.dump(results, f, indent=4)
