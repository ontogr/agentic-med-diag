// Read the ontology mirror written by register_ontology.
// Reserved __Onto labels keep this metadata out of Entity and RELATES_TO queries.
MATCH (m:`__OntoMeta` {key: 'ontology'})
RETURN m.version AS version, m.schema_json AS schema_json
