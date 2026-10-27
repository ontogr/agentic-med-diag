// Upsert allowed (head, type, tail) triples as ONTO_PAIR edges.
// Entity types must come first; a pair with an undeclared endpoint matches nothing.
UNWIND $records AS p
MATCH (h:`__OntoEntity` {label: p.head})
MATCH (t:`__OntoEntity` {label: p.tail})
MERGE (h)-[r:ONTO_PAIR {type: p.type}]->(t)
SET r.glirel_label = p.glirel_label, r.extractable = p.extractable
