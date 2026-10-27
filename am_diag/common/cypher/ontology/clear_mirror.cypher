// Remove ontology entity-type nodes and their ONTO_PAIR edges.
// Used before a fresh registration so records from an interrupted attempt do not remain.
MATCH (n:`__OntoEntity`)
DETACH DELETE n
