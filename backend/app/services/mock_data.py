"""
Literature retrieval layer.

Primary path:  arXiv API (live, no key required) — works for any topic.
Fallback path: hardcoded mock corpus — used when arXiv is unreachable or
               returns nothing (offline demo, rate-limited, unknown topic).

The format_context_block() output is identical regardless of which path ran,
so the NeMoClaw prompt is unaffected by the retrieval source.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

CORPUS: dict[str, list[dict]] = {
    "efficient llm routing": [
        {
            "title": "MoE-LLaVA: Mixture of Experts for Large Vision-Language Models",
            "authors": "Lin et al., 2024",
            "abstract": (
                "We propose MoE-LLaVA, a sparse mixture-of-experts (MoE) framework for "
                "multi-modal LLMs. By routing visual tokens to specialised expert FFN layers "
                "while keeping attention dense, we reduce active parameters by 40 % without "
                "degrading performance on MMMU or ScienceQA. A key limitation is that the "
                "routing decisions are made independently per token, ignoring cross-token "
                "semantic coherence — causing expert collapse under adversarial prompts."
            ),
        },
        {
            "title": "RouteLLM: Learning to Route LLMs with Preference Data",
            "authors": "Ong et al., 2024",
            "abstract": (
                "RouteLLM trains a lightweight classifier to decide whether a query should "
                "be served by a large or small LLM, saving 40–70 % of inference cost. "
                "However, the router is trained on static preference datasets and therefore "
                "fails to adapt when the domain distribution drifts at inference time. "
                "There is no mechanism for the router to signal uncertainty or abstain when "
                "a query falls outside its training distribution."
            ),
        },
        {
            "title": "Adaptive Inference via Speculative Decoding with Dynamic Draft Models",
            "authors": "Leviathan et al., 2023",
            "abstract": (
                "Speculative decoding dramatically reduces per-token latency by generating "
                "draft tokens with a small model and verifying them in parallel with a large "
                "model. The speedup is highly sensitive to the accept rate, which varies with "
                "input entropy. Current approaches use a fixed draft model and do not "
                "dynamically select among a pool of draft models based on estimated "
                "difficulty of the incoming sequence."
            ),
        },
        {
            "title": "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking",
            "authors": "Jiang et al., 2023",
            "abstract": (
                "LLM-Blender uses a PairRanker to select the best generation from an "
                "ensemble of models and a GenFuser to merge candidates into a higher-quality "
                "output. The system assumes each model generates a full response before "
                "ranking — incurring full inference cost for every model in the ensemble "
                "regardless of query difficulty. No early-exit mechanism exists."
            ),
        },
    ],
    "vision transformer robustness": [
        {
            "title": "How Robust are Vision Transformers to Common Corruptions?",
            "authors": "Naseer et al., 2021",
            "abstract": (
                "We benchmark ViT variants on ImageNet-C and find they are significantly "
                "more robust than CNNs to texture corruptions but remain brittle under "
                "frequency-based perturbations. The self-attention mechanism attends to "
                "global patches and loses fine-grained local feature sensitivity needed for "
                "detecting Gaussian noise at high severity. Positional encoding schemes "
                "are not evaluated under distribution shift."
            ),
        },
        {
            "title": "Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?",
            "authors": "Fu et al., 2022",
            "abstract": (
                "Patch-Fool proposes token-level adversarial perturbations targeting "
                "high-attention patches in ViTs. Despite ViT's claim of inherent robustness, "
                "Patch-Fool achieves >90 % ASR on DeiT. The attack exposes that ViTs rely "
                "heavily on a small subset of salient tokens. No existing defense adapts "
                "attention routing dynamically to neutralise token-level attacks."
            ),
        },
        {
            "title": "Improving Robustness of Vision Transformers by Reducing Sensitivity to Patch Corruptions",
            "authors": "Ming et al., 2022",
            "abstract": (
                "This work shows that masking corrupted patches during inference improves "
                "clean accuracy robustness under common corruptions. However, the corrupted "
                "patch identification relies on a separately trained binary classifier, "
                "making the pipeline fragile to unseen corruption types and adding 15 ms "
                "of detection latency that is not profiled under real-time constraints."
            ),
        },
    ],
    "llm hallucination detection": [
        {
            "title": "Selfcheckgpt: Zero-Resource Black-Box Hallucination Detection for LLMs",
            "authors": "Manakul et al., 2023",
            "abstract": (
                "SelfCheckGPT samples multiple responses from an LLM and measures "
                "consistency to estimate factual grounding without external knowledge. "
                "The method requires N inference passes per query (typically N=10–20), "
                "multiplying compute cost dramatically. It also cannot distinguish between "
                "consistent hallucinations that all sampled responses agree on — a "
                "systematic failure mode that SelfCheckGPT cannot detect by design."
            ),
        },
        {
            "title": "FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long-Form Generation",
            "authors": "Min et al., 2023",
            "abstract": (
                "FActScore decomposes generated text into atomic claims and verifies each "
                "against a reference knowledge base. It achieves high precision but requires "
                "a curated knowledge base, which limits applicability to open-domain queries. "
                "The claim decomposition step introduces its own error mode and does not "
                "propagate uncertainty across interdependent claims in a document."
            ),
        },
        {
            "title": "RAG vs. Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture",
            "authors": "Ovadia et al., 2023",
            "abstract": (
                "This empirical study compares RAG and fine-tuned LLMs on knowledge-intensive "
                "tasks. RAG reduces hallucination rate by 22 % over vanilla generation but "
                "the retrieved context itself can be factually inconsistent, and current "
                "systems do not model the trustworthiness of individual retrieved passages "
                "before injecting them into the prompt."
            ),
        },
    ],
    "protein structure prediction": [
        {
            "title": "Highly accurate protein structure prediction with AlphaFold",
            "authors": "Jumper et al., 2021",
            "abstract": (
                "AlphaFold2 achieves near-experimental accuracy on CASP14 by leveraging "
                "multiple sequence alignments (MSAs) and pair representations through "
                "Evoformer blocks. The model struggles with proteins lacking deep MSAs "
                "(orphan proteins or de-novo designed sequences) and cannot predict "
                "conformational ensembles — producing only a single structure per sequence."
            ),
        },
        {
            "title": "ESMFold: Language models of protein sequences at the scale of the universe",
            "authors": "Lin et al., 2023",
            "abstract": (
                "ESMFold achieves AlphaFold2-comparable accuracy without MSAs by using a "
                "protein language model for residue-level embeddings. Inference is 60× "
                "faster. However, pLDDT confidence scores are poorly calibrated for "
                "disordered regions and the model does not model flexibility or binding "
                "pocket conformational changes relevant to drug discovery."
            ),
        },
        {
            "title": "RoseTTAFold All-Atom: Unified modeling of protein-ligand, protein-DNA, and protein-protein interactions",
            "authors": "Krishna et al., 2024",
            "abstract": (
                "RoseTTAFold All-Atom extends protein structure prediction to model "
                "non-protein atoms enabling ligand binding pose prediction. Despite this "
                "advance, the model requires explicit covalent bond topology at inference "
                "time, limiting its applicability to novel covalent inhibitors where bond "
                "topology is the very question being investigated."
            ),
        },
    ],
}

# Keyword → corpus key mapping for fuzzy lookup
_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["routing", "router", "moe", "mixture", "speculative", "blender"], "efficient llm routing"),
    (["vit", "vision transformer", "patch", "corruption", "robust"], "vision transformer robustness"),
    (["hallucin", "factual", "rag", "retrieval", "selfcheck", "factscore"], "llm hallucination detection"),
    (["protein", "alphafold", "esmfold", "rosetta", "structure"], "protein structure prediction"),
]

_DEFAULT_CORPUS_KEY = "efficient llm routing"


def get_papers(topic: str, max_results: int = 4) -> tuple[list[dict], str]:
    """
    Return (papers_list, source_label) for the given topic string.

    Tries arXiv first — works for any topic, no key required.
    Falls back to the mock corpus if arXiv fails or is unreachable.
    source_label is "arxiv" or the matched mock corpus key.
    """
    try:
        from app.services.arxiv_client import fetch_from_arxiv
        papers = fetch_from_arxiv(topic, max_results=max_results)
        return papers, "arxiv"
    except Exception as exc:
        logger.warning(
            "arXiv retrieval failed for topic '%s' (%s: %s) — falling back to mock corpus.",
            topic, type(exc).__name__, exc,
        )

    # Mock fallback: keyword match → closest hardcoded corpus
    topic_lower = topic.lower()
    for keywords, corpus_key in _KEYWORD_MAP:
        if any(kw in topic_lower for kw in keywords):
            logger.info("Mock corpus matched: '%s'", corpus_key)
            return CORPUS[corpus_key], corpus_key

    logger.info("No mock corpus match — using default '%s'", _DEFAULT_CORPUS_KEY)
    return CORPUS[_DEFAULT_CORPUS_KEY], _DEFAULT_CORPUS_KEY


def get_available_topics() -> list[str]:
    """Expose hardcoded mock topics for API discoverability."""
    return list(CORPUS.keys())


def format_context_block(papers: list[dict]) -> str:
    """
    Render papers into the structured context block injected into the NeMoClaw prompt.
    Each paper is a clearly delimited block to help the LLM parse boundaries.
    """
    blocks = []
    for i, p in enumerate(papers, start=1):
        blocks.append(
            f"[PAPER {i}]\n"
            f"Title: {p['title']}\n"
            f"Authors: {p['authors']}\n"
            f"Abstract:\n{p['abstract']}"
        )
    return "\n\n---\n\n".join(blocks)
