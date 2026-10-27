"""Name-keyed registries for corpus loaders, graph loaders, QA datasets, and schemas.

The CLI resolves ``--source corpus:<name>``, ``--source loader:<name>:<dir>``,
``--datasets`` selections, and ``graphs create --schema`` against these registries.
"""

from __future__ import annotations

from collections.abc import Callable

from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA, OPTIMUSKG_SCHEMA
from am_diag.loaders.corpus.base import CorpusLoader
from am_diag.loaders.corpus.clinical_guidelines import ClinicalGuidelinesCorpusLoader
from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader
from am_diag.loaders.corpus.pubmed_case_reports import PubmedCaseReportsCorpusLoader
from am_diag.loaders.corpus.statpearls import StatPearlsCorpusLoader
from am_diag.loaders.corpus.textbooks import TextbooksCorpusLoader
from am_diag.loaders.dataset import (
    CareQADataset,
    CareQAReasoningDataset,
    HealthBenchDataset,
    MedCaseReasoningDataset,
    MedMCQADataset,
    MedQADataset,
    MedXpertQADataset,
    MMLUMedDataset,
    MMLUProHealthDataset,
    NEJMDiagnosticDataset,
    NEJMQADataset,
    PubHealthBenchDataset,
    PubHealthBenchFreeformDataset,
    PubMedQADataset,
    QADataset,
    RARMedDataset,
    SuperGPQAMedDataset,
)
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.bgee import BgeeGraphLoader
from am_diag.loaders.graph.ctd import CtdGraphLoader
from am_diag.loaders.graph.disgenet import DisGeNETGraphLoader
from am_diag.loaders.graph.drugbank import DrugBankGraphLoader
from am_diag.loaders.graph.drugcentral import DrugCentralGraphLoader
from am_diag.loaders.graph.gene_ontology import GeneOntologyGraphLoader
from am_diag.loaders.graph.onsides import OnsidesGraphLoader
from am_diag.loaders.graph.ontology_backbones import (
    HpoGraphLoader,
    MondoGraphLoader,
    UberonGraphLoader,
)
from am_diag.loaders.graph.opentargets import OpenTargetsGraphLoader
from am_diag.loaders.graph.optimuskg import OptimusKGGraphLoader
from am_diag.loaders.graph.ppi import PpiGraphLoader
from am_diag.loaders.graph.reactome import ReactomeGraphLoader


CORPUS_LOADER_REGISTRY: dict[str, type[CorpusLoader]] = {
    "clinical_guidelines": ClinicalGuidelinesCorpusLoader,
    "pubmed": PubMedCorpusLoader,
    "pubmed_case_reports": PubmedCaseReportsCorpusLoader,
    "statpearls": StatPearlsCorpusLoader,
    "textbooks": TextbooksCorpusLoader,
}

GRAPH_LOADER_REGISTRY: dict[str, Callable[..., GraphLoader]] = {
    "uberon": UberonGraphLoader,
    "mondo": MondoGraphLoader,
    "hpo": HpoGraphLoader,
    "gene_ontology": GeneOntologyGraphLoader,
    "reactome": ReactomeGraphLoader,
    "opentargets": OpenTargetsGraphLoader,
    "bgee": BgeeGraphLoader,
    "drugbank": DrugBankGraphLoader,
    "disgenet": DisGeNETGraphLoader,
    "drugcentral": DrugCentralGraphLoader,
    "onsides": OnsidesGraphLoader,
    "ctd": CtdGraphLoader,
    "ppi": PpiGraphLoader,
    "optimuskg": OptimusKGGraphLoader,
}

DATASET_REGISTRY: dict[str, type[QADataset]] = {
    "careqa": CareQADataset,
    "careqa_reasoning": CareQAReasoningDataset,
    "healthbench": HealthBenchDataset,
    "medcase_reasoning": MedCaseReasoningDataset,
    "medmcqa": MedMCQADataset,
    "medqa": MedQADataset,
    "medxpertqa": MedXpertQADataset,
    "mmlu_med": MMLUMedDataset,
    "mmlu_pro_health": MMLUProHealthDataset,
    "nejm_diagnostic": NEJMDiagnosticDataset,
    "nejm_qa": NEJMQADataset,
    "pubhealthbench": PubHealthBenchDataset,
    "pubhealthbench_freeform": PubHealthBenchFreeformDataset,
    "pubmedqa": PubMedQADataset,
    "rar_med": RARMedDataset,
    "supergpqa_med": SuperGPQAMedDataset,
}

SCHEMA_REGISTRY: dict[str, GraphSchema] = {
    "medical": MEDICAL_GRAPHRAG_SCHEMA,
    "optimuskg": OPTIMUSKG_SCHEMA,
}


__all__ = [
    "CORPUS_LOADER_REGISTRY",
    "DATASET_REGISTRY",
    "GRAPH_LOADER_REGISTRY",
    "SCHEMA_REGISTRY",
]
