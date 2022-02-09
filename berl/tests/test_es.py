from berl import *
import berl
import cma
import numpy as np


def get_cfg(preset=None):
    s = f"--preset {preset}".split() if preset is not None else ""
    args, unknown = parser.parse_known_args(s)
    args = load_preset(args)
    return args.__dict__


def es_test(es, n_genes, cfg):
    assert es.n_genes == n_genes
    es.populate()
    assert isinstance(es.genomes, (list, np.ndarray))
    assert len(es.genomes) == cfg["pop"]
    assert isinstance(es.genomes[0], (list, np.ndarray))

    pop = es.ask()
    assert isinstance(pop, (list, np.ndarray))
    assert len(pop) == cfg["pop"]
    assert all(all(pop[i] == es.genomes[i]) for i in range(len(pop)))

    one_max_test(es)


def one_max(x):
    return - np.sum(np.abs(x - 1))


def one_max_test(es):
    print(es)
    fitnesses = []
    for _ in range(1000):
        pop = es.ask()
        fit = [one_max(g) for g in pop]
        fitnesses.append(np.mean(fit))
        es.tell(pop, fit)
    print(f"{fitnesses[0]} -> {fitnesses[-1]}")
    assert fitnesses[0] < fitnesses[-1]  # Check if it maximizes the fitness


def test_canonical():
    cfg = get_cfg('canonical')
    n_genes = 10
    es = Canonical(n_genes, cfg)
    es_test(es, n_genes, cfg)


def test_openai():
    cfg = get_cfg('openai')
    n_genes = 10
    for g in ["base", "sgd", "adam"]:
        cfg["es_gradient"] = g
        es = OpenAI(n_genes, cfg)
        es_test(es, n_genes, cfg)


def test_snes():
    cfg = get_cfg()
    n_genes = 10
    es = SNES(n_genes, cfg)
    es_test(es, n_genes, cfg)


def test_cmaes():
    cfg = get_cfg()
    n_genes = 10
    es = CMAES(n_genes, cfg)
    es_test(es, n_genes, cfg)


if __name__ == "__main__":
    tests = [test_canonical, test_openai, test_snes]
    for t in tests:
        try:
            t()
        except AssertionError:
            print("Test didn't pass")
        print("")
