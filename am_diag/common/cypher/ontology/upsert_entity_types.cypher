// The reserved label keeps these nodes out of Entity queries.
UNWIND $records AS e
MERGE (n:`__OntoEntity` {label: e.label})
SET n.description = e.description,
    n.gliner_label = e.gliner_label,
    n.extractable = e.extractable
