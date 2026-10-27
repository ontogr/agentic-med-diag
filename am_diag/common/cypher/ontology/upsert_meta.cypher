// Record the ontology version, mode and full definition on the single meta node.
// The definition is kept as JSON so the schema can be rebuilt without a second read.
MERGE (m:`__OntoMeta` {key: 'ontology'})
SET m.version = $version, m.mode = $mode, m.schema_json = $schema_json
