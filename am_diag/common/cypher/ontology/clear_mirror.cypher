// Used before a fresh registration so records from an interrupted attempt do not remain.
MATCH (n:`__OntoEntity`)
DETACH DELETE n
